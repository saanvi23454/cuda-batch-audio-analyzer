#include "labels.h"

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <vector>

namespace audio {
namespace {

std::vector<std::string> SplitCsvLine(const std::string& line) {
  std::vector<std::string> fields;
  std::stringstream stream(line);
  std::string field;
  while (std::getline(stream, field, ',')) {
    if (!field.empty() && field.back() == '\r') field.pop_back();
    fields.push_back(field);
  }
  if (!line.empty() && line.back() == ',') fields.push_back("");
  return fields;
}

}  // namespace

std::map<std::string, Label> ReadLabels(const std::string& path) {
  std::map<std::string, Label> labels;
  std::ifstream file(path);
  if (!file) return labels;
  std::string line;
  std::getline(file, line);  // Header.
  while (std::getline(file, line)) {
    const std::vector<std::string> fields = SplitCsvLine(line);
    if (fields.size() < 4 || fields[0].empty()) continue;
    Label label;
    label.kind = fields[1];
    if (!fields[2].empty()) label.f0_hz = std::atof(fields[2].c_str());
    if (!fields[3].empty()) label.num_notes = std::atoi(fields[3].c_str());
    labels[fields[0]] = label;
  }
  return labels;
}

}  // namespace audio
