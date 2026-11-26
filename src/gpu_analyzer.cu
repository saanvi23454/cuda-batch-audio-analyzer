#include "gpu_analyzer.h"

#include <cuda_runtime.h>
#include <cufft.h>
#include <npp.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <sstream>

#include "cuda_check.h"
#include "kernels.h"

namespace audio {
namespace {

// cuFFT runs over the frames of a signal in chunks of this many frames, so
// one plan per lane serves signals of any length.
constexpr int kFftFramesPerChunk = 64;
// Dynamic range shown in the spectrogram images.
constexpr float kImageRangeDb = 80.0f;

int RoundUp(int value, int multiple) {
  return (value + multiple - 1) / multiple * multiple;
}

// The integer types of NPP's signal functions changed between CUDA versions
// (int in 11.x, size_t in newer releases). These helpers deduce them from
// the function pointer, so the same code builds against either.
template <typename LengthT, typename SizeT>
size_t QueryScratch(NppStatus (*query)(LengthT, SizeT*), size_t length) {
  SizeT bytes = 0;
  NPP_CHECK(query(static_cast<LengthT>(length), &bytes));
  return static_cast<size_t>(bytes);
}

template <typename LengthT, typename SizeT>
size_t QueryScratch(NppStatus (*query)(LengthT, SizeT*, NppStreamContext),
                    size_t length, NppStreamContext context) {
  SizeT bytes = 0;
  NPP_CHECK(query(static_cast<LengthT>(length), &bytes, context));
  return static_cast<size_t>(bytes);
}

// Scratch bytes needed by the four NPP reductions for |length| elements.
size_t NppScratchBytes(size_t length, NppStreamContext context) {
  size_t bytes = 0;
#if CUDART_VERSION >= 13000
  bytes = std::max(bytes, QueryScratch(nppsMeanGetBufferSize_32f_Ctx, length,
                                       context));
  bytes = std::max(bytes, QueryScratch(nppsMinMaxGetBufferSize_32f_Ctx,
                                       length, context));
  bytes = std::max(bytes, QueryScratch(nppsMaxGetBufferSize_32f_Ctx, length,
                                       context));
  bytes = std::max(bytes, QueryScratch(nppsMaxIndxGetBufferSize_32f_Ctx,
                                       length, context));
#else
  (void)context;
  bytes = std::max(bytes, QueryScratch(nppsMeanGetBufferSize_32f, length));
  bytes = std::max(bytes, QueryScratch(nppsMinMaxGetBufferSize_32f, length));
  bytes = std::max(bytes, QueryScratch(nppsMaxGetBufferSize_32f, length));
  bytes = std::max(bytes, QueryScratch(nppsMaxIndxGetBufferSize_32f, length));
#endif
  return bytes;
}

// Calls nppsMaxIndx_32f_Ctx, whose index type is int or size_t depending on
// the CUDA version. Stores the index size in |index_bytes| so the host can
// read the 8-byte result slot correctly.
template <typename LengthT, typename IndexT>
void MaxIndex(NppStatus (*max_index)(const Npp32f*, LengthT, Npp32f*,
                                     IndexT*, Npp8u*, NppStreamContext),
              const float* source, int length, float* max_value,
              void* index_slot, Npp8u* scratch, NppStreamContext context,
              int* index_bytes) {
  static_assert(sizeof(IndexT) <= 8, "index slot holds 8 bytes");
  *index_bytes = static_cast<int>(sizeof(IndexT));
  NPP_CHECK(max_index(source, static_cast<LengthT>(length), max_value,
                      static_cast<IndexT*>(index_slot), scratch, context));
}

NppStreamContext MakeNppContext(cudaStream_t stream, int device,
                                const cudaDeviceProp& properties) {
  NppStreamContext context;
  std::memset(&context, 0, sizeof(context));  // nStreamFlags = 0 matches
  context.hStream = stream;                   // cudaStreamCreate().
  context.nCudaDeviceId = device;
  context.nMultiProcessorCount = properties.multiProcessorCount;
  context.nMaxThreadsPerMultiProcessor =
      properties.maxThreadsPerMultiProcessor;
  context.nMaxThreadsPerBlock = properties.maxThreadsPerBlock;
  context.nSharedMemPerBlock = properties.sharedMemPerBlock;
  context.nCudaDevAttrComputeCapabilityMajor = properties.major;
  context.nCudaDevAttrComputeCapabilityMinor = properties.minor;
  return context;
}

}  // namespace

// Everything one stream needs to process one signal at a time.
struct Lane {
  cudaStream_t stream = nullptr;
  NppStreamContext npp;
  cufftHandle plan = 0;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;

  // Pinned host buffers (required for truly asynchronous copies).
  int16_t* host_samples = nullptr;
  float* host_features = nullptr;
  float* host_mean_spectrum = nullptr;
  uint8_t* host_peak_index = nullptr;  // 8-byte slot.
  uint8_t* host_image = nullptr;

  // Device buffers.
  int16_t* samples = nullptr;
  float* signal = nullptr;
  float* stats = nullptr;  // kNumStats scalars.
  float* frames = nullptr;
  cufftComplex* spectrum = nullptr;
  float* power = nullptr;
  float* db = nullptr;
  float* features = nullptr;
  float* mean_spectrum = nullptr;
  void* peak_index = nullptr;  // 8-byte slot.
  uint8_t* image = nullptr;
  Npp8u* scratch = nullptr;

  // The signal this lane is working on (-1 when idle).
  int pending = -1;
  int num_frames = 0;
  bool want_image = false;
};

struct GpuAnalyzer::Impl {
  StftConfig config;
  size_t max_samples = 0;
  int max_frames = 0;
  int device = 0;
  cudaDeviceProp properties;
  int peak_index_bytes = 4;
  std::vector<Lane> lanes;

  void CreateLane(Lane* lane);
  void DestroyLane(Lane* lane);
  void Enqueue(Lane* lane, int index, const Signal& signal, bool want_image);
  void Collect(Lane* lane, std::vector<SignalAnalysis>* results);
};

void GpuAnalyzer::Impl::CreateLane(Lane* lane) {
  const int bins = config.NumBins();
  const int padded_frames = RoundUp(max_frames, kFftFramesPerChunk);
  const size_t spectrogram_values = static_cast<size_t>(max_frames) * bins;

  CUDA_CHECK(cudaStreamCreate(&lane->stream));
  CUDA_CHECK(cudaEventCreate(&lane->start));
  CUDA_CHECK(cudaEventCreate(&lane->stop));
  lane->npp = MakeNppContext(lane->stream, device, properties);

  // One batched R2C plan: kFftFramesPerChunk transforms of frame_length
  // real samples each, with the default packed layout.
  int n = config.frame_length;
  CUFFT_CHECK(cufftPlanMany(&lane->plan, 1, &n, nullptr, 1, 0, nullptr, 1, 0,
                            CUFFT_R2C, kFftFramesPerChunk));
  CUFFT_CHECK(cufftSetStream(lane->plan, lane->stream));

  CUDA_CHECK(cudaMallocHost(&lane->host_samples,
                            max_samples * sizeof(int16_t)));
  CUDA_CHECK(cudaMallocHost(&lane->host_features,
                            sizeof(float) * kNumFeatures * max_frames));
  CUDA_CHECK(cudaMallocHost(&lane->host_mean_spectrum, sizeof(float) * bins));
  CUDA_CHECK(cudaMallocHost(&lane->host_peak_index, 8));
  CUDA_CHECK(cudaMallocHost(&lane->host_image, spectrogram_values));

  CUDA_CHECK(cudaMalloc(&lane->samples, max_samples * sizeof(int16_t)));
  CUDA_CHECK(cudaMalloc(&lane->signal, max_samples * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&lane->stats, kNumStats * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&lane->frames, sizeof(float) * padded_frames *
                                           config.frame_length));
  CUDA_CHECK(cudaMalloc(&lane->spectrum,
                        sizeof(cufftComplex) * padded_frames * bins));
  CUDA_CHECK(cudaMalloc(&lane->power, sizeof(float) * spectrogram_values));
  CUDA_CHECK(cudaMalloc(&lane->db, sizeof(float) * spectrogram_values));
  CUDA_CHECK(cudaMalloc(&lane->features,
                        sizeof(float) * kNumFeatures * max_frames));
  CUDA_CHECK(cudaMalloc(&lane->mean_spectrum, sizeof(float) * bins));
  CUDA_CHECK(cudaMalloc(&lane->peak_index, 8));
  CUDA_CHECK(cudaMalloc(&lane->image, spectrogram_values));

  // The NPP reductions run on the signal, the spectrogram and the mean
  // spectrum; one scratch buffer sized for the longest of them is shared,
  // which is safe because they run one after another on this stream.
  const size_t longest = std::max(max_samples, spectrogram_values);
  const size_t scratch_bytes = std::max<size_t>(
      1, NppScratchBytes(longest, lane->npp));
  CUDA_CHECK(cudaMalloc(&lane->scratch, scratch_bytes));
}

void GpuAnalyzer::Impl::DestroyLane(Lane* lane) {
  cufftDestroy(lane->plan);
  cudaFreeHost(lane->host_samples);
  cudaFreeHost(lane->host_features);
  cudaFreeHost(lane->host_mean_spectrum);
  cudaFreeHost(lane->host_peak_index);
  cudaFreeHost(lane->host_image);
  cudaFree(lane->samples);
  cudaFree(lane->signal);
  cudaFree(lane->stats);
  cudaFree(lane->frames);
  cudaFree(lane->spectrum);
  cudaFree(lane->power);
  cudaFree(lane->db);
  cudaFree(lane->features);
  cudaFree(lane->mean_spectrum);
  cudaFree(lane->peak_index);
  cudaFree(lane->image);
  cudaFree(lane->scratch);
  cudaEventDestroy(lane->start);
  cudaEventDestroy(lane->stop);
  cudaStreamDestroy(lane->stream);
}

// Queues the whole pipeline for one signal on the lane's stream. Nothing
// here waits for the GPU, so the host can move on to the next lane.
void GpuAnalyzer::Impl::Enqueue(Lane* lane, int index, const Signal& signal,
                                bool want_image) {
  const int n = static_cast<int>(signal.samples.size());
  const int frame_length = config.frame_length;
  const int bins = config.NumBins();
  const int frames = config.NumFrames(signal.samples.size());
  const int padded_frames = RoundUp(frames, kFftFramesPerChunk);
  const size_t spectrogram_values = static_cast<size_t>(frames) * bins;
  cudaStream_t stream = lane->stream;

  lane->pending = index;
  lane->num_frames = frames;
  lane->want_image = want_image;
  std::memcpy(lane->host_samples, signal.samples.data(),
              sizeof(int16_t) * n);

  CUDA_CHECK(cudaEventRecord(lane->start, stream));
  CUDA_CHECK(cudaMemcpyAsync(lane->samples, lane->host_samples,
                             sizeof(int16_t) * n, cudaMemcpyHostToDevice,
                             stream));

  // Stage 1 (NPP): int16 -> float, then mean / min / max, then a kernel
  // removes the DC offset and scales the peak to 1.
  NPP_CHECK(nppsConvert_16s32f_Ctx(lane->samples, lane->signal, n,
                                   lane->npp));
  NPP_CHECK(nppsMean_32f_Ctx(lane->signal, n, lane->stats + kStatMean,
                             NPP_ALG_HINT_NONE, lane->scratch, lane->npp));
  NPP_CHECK(nppsMinMax_32f_Ctx(lane->signal, n, lane->stats + kStatMin,
                               lane->stats + kStatMax, lane->scratch,
                               lane->npp));
  LaunchNormalize(lane->signal, n, lane->stats, stream);

  // Stage 2: Hann-windowed frames -> batched cuFFT -> power and dB.
  LaunchFrameAndWindow(lane->signal, n, frame_length, config.hop, frames,
                       padded_frames, lane->frames, stream);
  for (int chunk = 0; chunk < padded_frames; chunk += kFftFramesPerChunk) {
    CUFFT_CHECK(cufftExecR2C(
        lane->plan, lane->frames + static_cast<size_t>(chunk) * frame_length,
        lane->spectrum + static_cast<size_t>(chunk) * bins));
  }
  const float power_scale =
      1.0f / (static_cast<float>(frame_length) * frame_length);
  LaunchPowerSpectrum(lane->spectrum, static_cast<int>(spectrogram_values),
                      power_scale, lane->power, lane->db, stream);

  // Stage 3: per-frame features and the mean spectrum.
  const float bin_hz = static_cast<float>(config.BinHz(signal.sample_rate));
  LaunchFrameFeatures(lane->signal, n, lane->power, frames, frame_length,
                      config.hop, bins, bin_hz, lane->features, stream);
  LaunchMeanSpectrum(lane->power, frames, bins, lane->mean_spectrum, stream);

  // Stage 4 (NPP): strongest frequency bin, skipping DC.
  MaxIndex(nppsMaxIndx_32f_Ctx, lane->mean_spectrum + 1, bins - 1,
           lane->stats + kStatPeakPower, lane->peak_index, lane->scratch,
           lane->npp, &peak_index_bytes);

  // Optional stage 5 (NPP + kernel): 8-bit spectrogram image.
  if (want_image) {
    NPP_CHECK(nppsMax_32f_Ctx(lane->db, static_cast<int>(spectrogram_values),
                              lane->stats + kStatDbMax, lane->scratch,
                              lane->npp));
    LaunchSpectrogramImage(lane->db, frames, bins, lane->stats, kImageRangeDb,
                           lane->image, stream);
    CUDA_CHECK(cudaMemcpyAsync(lane->host_image, lane->image,
                               spectrogram_values, cudaMemcpyDeviceToHost,
                               stream));
  }

  CUDA_CHECK(cudaMemcpyAsync(lane->host_features, lane->features,
                             sizeof(float) * kNumFeatures * frames,
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(lane->host_mean_spectrum, lane->mean_spectrum,
                             sizeof(float) * bins, cudaMemcpyDeviceToHost,
                             stream));
  CUDA_CHECK(cudaMemcpyAsync(lane->host_peak_index, lane->peak_index, 8,
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaEventRecord(lane->stop, stream));
}

// Waits for the lane's signal and copies its results out of pinned memory.
void GpuAnalyzer::Impl::Collect(Lane* lane,
                                std::vector<SignalAnalysis>* results) {
  CUDA_CHECK(cudaEventSynchronize(lane->stop));
  SignalAnalysis& result = (*results)[lane->pending];
  const int frames = lane->num_frames;
  const int bins = config.NumBins();

  CUDA_CHECK(cudaEventElapsedTime(&result.gpu_ms, lane->start, lane->stop));
  result.num_frames = frames;
  result.features.assign(lane->host_features,
                         lane->host_features + kNumFeatures * frames);
  result.mean_spectrum.assign(lane->host_mean_spectrum,
                              lane->host_mean_spectrum + bins);

  long long index = 0;
  if (peak_index_bytes == 8) {
    int64_t value = 0;
    std::memcpy(&value, lane->host_peak_index, 8);
    index = value;
  } else {
    int32_t value = 0;
    std::memcpy(&value, lane->host_peak_index, 4);
    index = value;
  }
  result.peak_bin = static_cast<int>(index) + 1;  // The search skipped DC.

  if (lane->want_image) {
    const size_t pixels = static_cast<size_t>(frames) * bins;
    result.spectrogram.assign(lane->host_image, lane->host_image + pixels);
  }
  lane->pending = -1;
}

GpuAnalyzer::GpuAnalyzer(const StftConfig& config, size_t max_samples,
                         int num_streams)
    : impl_(new Impl) {
  impl_->config = config;
  impl_->max_samples = std::max<size_t>(max_samples, 1);
  impl_->max_frames = config.NumFrames(impl_->max_samples);
  CUDA_CHECK(cudaGetDevice(&impl_->device));
  CUDA_CHECK(cudaGetDeviceProperties(&impl_->properties, impl_->device));
  impl_->lanes.resize(std::max(1, num_streams));
  for (Lane& lane : impl_->lanes) impl_->CreateLane(&lane);
  CUDA_CHECK(cudaDeviceSynchronize());
}

GpuAnalyzer::~GpuAnalyzer() {
  cudaDeviceSynchronize();
  for (Lane& lane : impl_->lanes) impl_->DestroyLane(&lane);
}

void GpuAnalyzer::AnalyzeBatch(const std::vector<Signal>& signals,
                               const std::vector<bool>& want_image,
                               std::vector<SignalAnalysis>* results) {
  results->assign(signals.size(), SignalAnalysis());
  std::vector<Lane>& lanes = impl_->lanes;
  // Round-robin over the lanes. Before a lane takes a new signal, the host
  // collects its previous one; meanwhile the other lanes keep the GPU busy.
  for (size_t i = 0; i < signals.size(); ++i) {
    Lane& lane = lanes[i % lanes.size()];
    if (lane.pending >= 0) impl_->Collect(&lane, results);
    const bool image = i < want_image.size() && want_image[i];
    impl_->Enqueue(&lane, static_cast<int>(i), signals[i], image);
  }
  for (Lane& lane : lanes) {
    if (lane.pending >= 0) impl_->Collect(&lane, results);
  }
}

std::string GpuAnalyzer::DeviceDescription() const {
  const cudaDeviceProp& p = impl_->properties;
  int runtime = 0;
  int driver = 0;
  int cufft = 0;
  CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
  CUDA_CHECK(cudaDriverGetVersion(&driver));
  CUFFT_CHECK(cufftGetVersion(&cufft));
  const NppLibraryVersion* npp = nppGetLibVersion();

  std::ostringstream out;
  out << "GPU " << impl_->device << ": " << p.name << " (compute capability "
      << p.major << "." << p.minor << ", " << p.multiProcessorCount
      << " SMs, " << (p.totalGlobalMem >> 20) << " MiB)\n"
      << "CUDA runtime " << runtime / 1000 << "." << (runtime % 1000) / 10
      << ", driver " << driver / 1000 << "." << (driver % 1000) / 10
      << ", cuFFT " << cufft << ", NPP " << npp->major << "." << npp->minor
      << "." << npp->build << "\n"
      << "Streams: " << impl_->lanes.size()
      << ", FFT chunk: " << kFftFramesPerChunk << " frames";
  return out.str();
}

}  // namespace audio
