#include <torch/csrc/jit/passes/pass_manager.h>

namespace torch {
namespace jit {

RegisterPostFusionPass::RegisterPostFusionPass(Pass p){
  registerPostFusionPass(p);
}

RegisterPreFusionPass::RegisterPreFusionPass(Pass p){
  registerPreFusionPass(p);
}

std::vector<PassEntry>& getCustomPostFusionPasses() {
  static std::vector<PassEntry> passes;
  return passes;
}

std::vector<PassEntry>& getCustomPreFusionPasses() {
  static std::vector<PassEntry> passes;
  return passes;
}

PassNameType RegisterPostFusionPass::registerPostFusionPass(Pass p) {
  getCustomPostFusionPasses().emplace_back(PassEntry{std::move(p), passID});
  return passID++;
}

PassNameType RegisterPreFusionPass::registerPreFusionPass(Pass p) {
  getCustomPreFusionPasses().emplace_back(PassEntry{std::move(p), passID});
  return passID++;
}

ClearPostFusionPass::ClearPostFusionPass(PassNameType pid) {
  auto& passes = getCustomPostFusionPasses();
  auto it = passes.begin();
  for (; it != passes.end(); it++) {
    if (pid == (*it).second)
      break;
  }
  if (it != passes.end())
    passes.erase(it);
}

ClearPreFusionPass::ClearPreFusionPass(PassNameType pid) {
  auto& passes = getCustomPreFusionPasses();
  auto it = passes.begin();
  for (; it != passes.end(); it++) {
    if (pid == (*it).second)
      break;
  }
  if (it != passes.end())
    passes.erase(it);
}

ClearAllPostFusionPasses::ClearAllPostFusionPasses() {
  auto& passes = getCustomPostFusionPasses();
  passes.erase(passes.begin(), passes.end());
}

ClearAllPreFusionPasses::ClearAllPreFusionPasses() {
  auto& passes = getCustomPreFusionPasses();
  passes.erase(passes.begin(), passes.end());
}

PassNameType PassManager::name(PassNameType PassName, bool set){
  static PassNameType name = 0;
  if(set)
    name = PassName;
  return name;
}

bool PassManager::flipRegistered(bool flip){
  static bool val = false;
  if(flip) val = !val;
  return val;
}
void PassManager::registerPass(Pass pass) {
  if (!flipRegistered()) {
    name( RegisterPostFusionPass::registerPostFusionPass(pass), true );
    flipRegistered(true);
  }
}

void PassManager::clearPass() {
  if (flipRegistered()) {
    ClearPostFusionPass pass(name());
    flipRegistered(true);
  }
}


} // namespace jit
} // namespace torch
