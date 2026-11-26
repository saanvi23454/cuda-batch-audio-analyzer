// Small file-system helpers (POSIX dirent, no C++17 <filesystem> needed).

#ifndef SRC_FILE_UTIL_H_
#define SRC_FILE_UTIL_H_

#include <cstdint>
#include <string>
#include <vector>

namespace audio {

// Returns the sorted names (not paths) of the files in |directory| whose
// extension matches |extension| (e.g. ".wav"), ignoring case.
std::vector<std::string> ListFiles(const std::string& directory,
                                   const std::string& extension);

// Creates |path| and any missing parent directories. Returns false on error.
bool MakeDirectories(const std::string& path);

std::string JoinPath(const std::string& directory, const std::string& name);

// "clip_01.wav" -> "clip_01".
std::string StripExtension(const std::string& name);

// Writes an 8-bit binary PGM (P5) image.
bool WritePgm(const std::string& path, int width, int height,
              const std::vector<uint8_t>& pixels);

}  // namespace audio

#endif  // SRC_FILE_UTIL_H_
