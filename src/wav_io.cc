#include "wav_io.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <vector>

namespace audio {
namespace {

constexpr uint16_t kFormatPcm = 1;
constexpr uint16_t kFormatExtensible = 0xFFFE;

uint16_t ReadU16(const uint8_t* p) {
  return static_cast<uint16_t>(p[0] | (p[1] << 8));
}

uint32_t ReadU32(const uint8_t* p) {
  return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
         (static_cast<uint32_t>(p[2]) << 16) |
         (static_cast<uint32_t>(p[3]) << 24);
}

void AppendU16(uint16_t v, std::vector<uint8_t>* out) {
  out->push_back(static_cast<uint8_t>(v & 0xFF));
  out->push_back(static_cast<uint8_t>(v >> 8));
}

void AppendU32(uint32_t v, std::vector<uint8_t>* out) {
  for (int i = 0; i < 4; ++i) {
    out->push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xFF));
  }
}

void AppendTag(const char* tag, std::vector<uint8_t>* out) {
  out->insert(out->end(), tag, tag + 4);
}

std::string BaseName(const std::string& path) {
  const size_t slash = path.find_last_of("/\\");
  return slash == std::string::npos ? path : path.substr(slash + 1);
}

}  // namespace

bool ReadWav(const std::string& path, Signal* signal, std::string* error) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    *error = "cannot open " + path;
    return false;
  }
  const std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(file)),
                                   std::istreambuf_iterator<char>());
  if (bytes.size() < 12 || std::memcmp(bytes.data(), "RIFF", 4) != 0 ||
      std::memcmp(bytes.data() + 8, "WAVE", 4) != 0) {
    *error = path + " is not a RIFF/WAVE file";
    return false;
  }

  uint16_t format = 0;
  uint16_t channels = 0;
  uint32_t sample_rate = 0;
  uint16_t bits = 0;
  size_t data_offset = 0;
  size_t data_size = 0;
  bool have_format = false;
  bool have_data = false;

  size_t pos = 12;
  while (pos + 8 <= bytes.size()) {
    const uint8_t* chunk = bytes.data() + pos;
    const size_t chunk_size = ReadU32(chunk + 4);
    const size_t body = pos + 8;
    const size_t available = bytes.size() - body;
    if (std::memcmp(chunk, "fmt ", 4) == 0 && chunk_size >= 16 &&
        available >= 16) {
      format = ReadU16(chunk + 8);
      channels = ReadU16(chunk + 10);
      sample_rate = ReadU32(chunk + 12);
      bits = ReadU16(chunk + 22);
      if (format == kFormatExtensible && chunk_size >= 26 &&
          available >= 26) {
        format = ReadU16(chunk + 8 + 24);  // First 2 bytes of the sub-format.
      }
      have_format = true;
    } else if (std::memcmp(chunk, "data", 4) == 0) {
      data_offset = body;
      // Some writers leave the size at 0 or too large; clamp to the file.
      data_size = (chunk_size == 0 || chunk_size > available) ? available
                                                              : chunk_size;
      have_data = true;
    }
    if (have_format && have_data) break;
    pos = body + chunk_size + (chunk_size & 1);  // Chunks are word aligned.
  }

  if (!have_format || !have_data) {
    *error = path + " has no fmt or data chunk";
    return false;
  }
  if (format != kFormatPcm || bits != 16 || channels == 0 ||
      sample_rate == 0) {
    *error = path + " is not 16-bit PCM (format " + std::to_string(format) +
             ", " + std::to_string(bits) + " bits)";
    return false;
  }

  const size_t frame_bytes = 2u * channels;
  const size_t num_samples = data_size / frame_bytes;
  if (num_samples == 0) {
    *error = path + " contains no samples";
    return false;
  }

  signal->name = BaseName(path);
  signal->sample_rate = static_cast<int>(sample_rate);
  signal->samples.resize(num_samples);
  const uint8_t* data = bytes.data() + data_offset;
  for (size_t i = 0; i < num_samples; ++i) {
    int32_t sum = 0;
    for (int c = 0; c < channels; ++c) {
      sum += static_cast<int16_t>(ReadU16(data + i * frame_bytes + 2 * c));
    }
    signal->samples[i] = static_cast<int16_t>(sum / channels);
  }
  return true;
}

bool WriteWav(const std::string& path, const Signal& signal,
              std::string* error) {
  const uint32_t data_bytes =
      static_cast<uint32_t>(signal.samples.size() * sizeof(int16_t));
  std::vector<uint8_t> out;
  out.reserve(44 + data_bytes);
  AppendTag("RIFF", &out);
  AppendU32(36 + data_bytes, &out);
  AppendTag("WAVE", &out);
  AppendTag("fmt ", &out);
  AppendU32(16, &out);
  AppendU16(kFormatPcm, &out);
  AppendU16(1, &out);  // Mono.
  AppendU32(static_cast<uint32_t>(signal.sample_rate), &out);
  AppendU32(static_cast<uint32_t>(signal.sample_rate) * 2, &out);
  AppendU16(2, &out);   // Block align.
  AppendU16(16, &out);  // Bits per sample.
  AppendTag("data", &out);
  AppendU32(data_bytes, &out);
  for (int16_t s : signal.samples) AppendU16(static_cast<uint16_t>(s), &out);

  std::ofstream file(path, std::ios::binary);
  if (!file) {
    *error = "cannot write " + path;
    return false;
  }
  file.write(reinterpret_cast<const char*>(out.data()),
             static_cast<std::streamsize>(out.size()));
  return static_cast<bool>(file);
}

}  // namespace audio
