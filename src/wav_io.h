// Minimal reader and writer for uncompressed 16-bit PCM WAV files.

#ifndef SRC_WAV_IO_H_
#define SRC_WAV_IO_H_

#include <string>

#include "signal_types.h"

namespace audio {

// Reads a 16-bit PCM WAV file (plain or WAVE_FORMAT_EXTENSIBLE). Files with
// several channels are mixed down to mono by averaging. Returns false and
// fills |error| when the file cannot be read or uses another format.
bool ReadWav(const std::string& path, Signal* signal, std::string* error);

// Writes |signal| as a mono 16-bit PCM WAV file.
bool WriteWav(const std::string& path, const Signal& signal,
              std::string* error);

}  // namespace audio

#endif  // SRC_WAV_IO_H_
