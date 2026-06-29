#pragma once

#include "model.h"

void yalm_coop_prepare(Model& model);
void yalm_coop_forward(Model& model, InferenceState& s, int token, int pos, InferenceMode mode);
