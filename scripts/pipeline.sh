#!/bin/bash
# Pipeline script to run the full pipeline stages sequentially.
# Each stage checks if its output directory exists to avoid redundant computation.

# Default args
if [[ -z "${OUT_DIR}" ]]; then
  OUT_DIR="outputs"
fi

if [[ -z "${PRED_OUT_DIR}" ]]; then
  PRED_OUT_DIR="outputs"
fi

if [[ -z "${SCENE_TYPE}" ]]; then
  SCENE_TYPE="indoor_synthetic"
fi

if [[ -z "${SCENE_NAME}" ]]; then
  SCENE_NAME="kitchen"
fi

if [[ -z "${SLF_ARGS}" ]]; then
  SLF_ARGS=""
fi

if [[ -z "${PRED_ARGS}" ]]; then
  PRED_ARGS=""
fi

if [[ -z "${AGGREGATE_SEGMENTATION_ARGS}" ]]; then
  AGGREGATE_SEGMENTATION_ARGS=""
fi

if [[ -z "${AGGREGATE_PRED_ARGS}" ]]; then
  AGGREGATE_PRED_ARGS=""
fi

if [[ -z "${EMITTER_INIT_ARGS}" ]]; then
  EMITTER_INIT_ARGS=""
fi

if [[ -z "${EMITTER_OPTIMIZATION_ARGS}" ]]; then
  EMITTER_OPTIMIZATION_ARGS=""
fi

if [[ -z "${BRDF_OPTIMIZATION_ARGS}" ]]; then
  BRDF_OPTIMIZATION_ARGS=""
fi

if [[ -z "${RENDER_ARGS}" ]]; then
  RENDER_ARGS=""
fi

echo "Output Directory: $OUT_DIR"
echo "Scene: $SCENE_TYPE/$SCENE_NAME"
echo "----------------------------------------------------------------"

# Stage 0: Baking SLF and calculating scene bbox
if [ ! -d "$OUT_DIR/$SCENE_TYPE/$SCENE_NAME/train/0_baking_slf/v0_iris_init" ]; then
  echo "$(date) Baking SLF..."
  LOG_FILE="$OUT_DIR/$SCENE_TYPE/$SCENE_NAME/train/0_baking_slf/v0_iris_init/logs.log"
  mkdir -p "$(dirname "$LOG_FILE")"
  COMMAND="python -m iif.job task=0_baking_slf/v0_iris_init component/scene@task.scene=$SCENE_TYPE/$SCENE_NAME paths.out_name=$OUT_DIR $SLF_ARGS"
  echo "$COMMAND" | tee -a "$LOG_FILE"
  eval "$COMMAND" 2>&1 | tee -a "$LOG_FILE"
else
    echo "$(date) SLF already baked, skipping..."
fi

echo "----------------------------------------------------------------"

# Stage 1: Single-View Prediction
if [ ! -d "$PRED_OUT_DIR/$SCENE_TYPE/$SCENE_NAME/train/1_single_view_prediction/v2_rgbx" ]; then
  echo "$(date) Running Single-View Prediction..."
  LOG_FILE="$PRED_OUT_DIR/$SCENE_TYPE/$SCENE_NAME/train/1_single_view_prediction/v2_rgbx/logs.log"
  mkdir -p "$(dirname "$LOG_FILE")"
  COMMAND="python -m iif.job task=1_single_view_prediction/v2_rgbx component/scene@task.scene=$SCENE_TYPE/$SCENE_NAME paths.out_name=$PRED_OUT_DIR $PRED_ARGS"
  echo "$COMMAND" | tee -a "$LOG_FILE"
  eval "$COMMAND" 2>&1 | tee -a "$LOG_FILE"
else
    echo "$(date) Single-View Prediction already done, skipping..."
fi

if [ "$PRED_OUT_DIR" != "$OUT_DIR" ] && [ ! -d "$OUT_DIR/$SCENE_TYPE/$SCENE_NAME/train/1_single_view_prediction/v2_rgbx" ]; then
    mkdir -p "$OUT_DIR/$SCENE_TYPE/$SCENE_NAME/train/1_single_view_prediction"
    ln -s "$(realpath "$PRED_OUT_DIR/$SCENE_TYPE/$SCENE_NAME/train/1_single_view_prediction/v2_rgbx")" "$(realpath "$OUT_DIR/$SCENE_TYPE/$SCENE_NAME/train/1_single_view_prediction/v2_rgbx")"
  fi

echo "----------------------------------------------------------------"

# Stage 2.1: Semantic Aggregation
if [ ! -d "$OUT_DIR/$SCENE_TYPE/$SCENE_NAME/train/2_aggregation/segmentation/v1_gt" ]; then
    echo "$(date) Running Semantic Aggregation..."
    LOG_FILE="$OUT_DIR/$SCENE_TYPE/$SCENE_NAME/train/2_aggregation/segmentation/v1_gt/logs.log"
    mkdir -p "$(dirname "$LOG_FILE")"
    COMMAND="python -m iif.job task=2_aggregation/segmentation/v1_gt component/scene@task.scene=$SCENE_TYPE/$SCENE_NAME paths.out_name=$OUT_DIR $AGGREGATE_SEGMENTATION_ARGS"
    echo "$COMMAND" | tee -a "$LOG_FILE"
    eval "$COMMAND" 2>&1 | tee -a "$LOG_FILE"
else
    echo "$(date) Semantic Aggregation already done, skipping..."
fi

echo "----------------------------------------------------------------"

# Stage 2.2: Aggregation
if [ ! -d "$OUT_DIR/$SCENE_TYPE/$SCENE_NAME/train/2_aggregation/v5_cm_soft_fullmat" ]; then
    echo "$(date) Running Aggregation..."
    LOG_FILE="$OUT_DIR/$SCENE_TYPE/$SCENE_NAME/train/2_aggregation/v5_cm_soft_fullmat/logs.log"
    mkdir -p "$(dirname "$LOG_FILE")"
    COMMAND="python -m iif.job task=2_aggregation/v5_cm_soft_fullmat component/scene@task.scene=$SCENE_TYPE/$SCENE_NAME paths.out_name=$OUT_DIR $AGGREGATE_PRED_ARGS"
    echo "$COMMAND" | tee -a "$LOG_FILE"
    eval "$COMMAND" 2>&1 | tee -a "$LOG_FILE"
else
    echo "$(date) Aggregation already done, skipping..."
fi

echo "----------------------------------------------------------------"

# Stage 3.1: Emitter Initialization
if [ ! -d "$OUT_DIR/$SCENE_TYPE/$SCENE_NAME/train/3_get_emitter/v0_iris_init" ]; then
    echo "$(date) Running Emitter Initialization..."
    LOG_FILE="$OUT_DIR/$SCENE_TYPE/$SCENE_NAME/train/3_get_emitter/v0_iris_init/logs.log"
    mkdir -p "$(dirname "$LOG_FILE")"
    COMMAND="python -m iif.job task=3_get_emitter/v0_iris_init component/scene@task.scene=$SCENE_TYPE/$SCENE_NAME paths.out_name=$OUT_DIR $EMITTER_INIT_ARGS"
    echo "$COMMAND" | tee -a "$LOG_FILE"
    eval "$COMMAND" 2>&1 | tee -a "$LOG_FILE"
else 
    echo "$(date) Emitter Initialization already done, skipping..."
fi

echo "----------------------------------------------------------------"

# Stage 3.2: Emitter Refinement
if [ ! -d "$OUT_DIR/$SCENE_TYPE/$SCENE_NAME/train/3_inverse_rendering/lighting/v0_2_iris_ldr" ]; then
    echo "$(date) Running Emitter Refinement..."
    LOG_FILE="$OUT_DIR/$SCENE_TYPE/$SCENE_NAME/train/3_inverse_rendering/lighting/v0_2_iris_ldr/logs.log"
    mkdir -p "$(dirname "$LOG_FILE")"
    COMMAND="python -m iif.job task=3_inverse_rendering/lighting/v0_2_iris_ldr component/scene@task.scene=$SCENE_TYPE/$SCENE_NAME paths.out_name=$OUT_DIR $EMITTER_OPTIMIZATION_ARGS"
    echo "$COMMAND" | tee -a "$LOG_FILE"
    eval "$COMMAND" 2>&1 | tee -a "$LOG_FILE"
else
    echo "$(date) Emitter Refinement already done, skipping..."
fi

echo "----------------------------------------------------------------"

# Stage 3.3: Shading Caching
if [ ! -d "$OUT_DIR/$SCENE_TYPE/$SCENE_NAME/train/3_inverse_rendering/shading/v0_2_iris_ldr" ]; then
    echo "$(date) Running Shading Caching..."
    LOG_FILE="$OUT_DIR/$SCENE_TYPE/$SCENE_NAME/train/3_inverse_rendering/shading/v0_2_iris_ldr/logs.log"
    mkdir -p "$(dirname "$LOG_FILE")"
    COMMAND="python -m iif.job task=3_inverse_rendering/shading/v0_2_iris_ldr component/scene@task.scene=$SCENE_TYPE/$SCENE_NAME paths.out_name=$OUT_DIR $EMITTER_OPTIMIZATION_ARGS"
    echo "$COMMAND" | tee -a "$LOG_FILE"
    eval "$COMMAND" 2>&1 | tee -a "$LOG_FILE"
else
    echo "$(date) Shading Caching already done, skipping..."
fi

echo "----------------------------------------------------------------"

# Stage 3.4: BRDF+CRF Refinement
if [ ! -d "$OUT_DIR/$SCENE_TYPE/$SCENE_NAME/train/3_inverse_rendering/material/v0_2_iris_ldr" ]; then
    echo "$(date) Running BRDF+CRF Refinement..."
    LOG_FILE="$OUT_DIR/$SCENE_TYPE/$SCENE_NAME/train/3_inverse_rendering/material/v0_2_iris_ldr/logs.log"
    mkdir -p "$(dirname "$LOG_FILE")"
    COMMAND="python -m iif.job task=3_inverse_rendering/material/v0_2_iris_ldr component/scene@task.scene=$SCENE_TYPE/$SCENE_NAME paths.out_name=$OUT_DIR $BRDF_OPTIMIZATION_ARGS"
    echo "$COMMAND" | tee -a "$LOG_FILE"
    eval "$COMMAND" 2>&1 | tee -a "$LOG_FILE"
else
    echo "$(date) BRDF+CRF Refinement already done, skipping..."
fi

echo "----------------------------------------------------------------"

# Stage 4: Rendering
if [ ! -d "$OUT_DIR/$SCENE_TYPE/$SCENE_NAME/train/4_render/v0_render" ]; then
    echo "$(date) Running Rendering..."
    LOG_FILE="$OUT_DIR/$SCENE_TYPE/$SCENE_NAME/train/4_render/v0_render/logs.log"
    mkdir -p "$(dirname "$LOG_FILE")"
    COMMAND="python -m iif.job task=4_render/v0_render component/scene@task.scene=$SCENE_TYPE/$SCENE_NAME paths.out_name=$OUT_DIR $RENDER_ARGS"
    echo "$COMMAND" | tee -a "$LOG_FILE"
    eval "$COMMAND" 2>&1 | tee -a "$LOG_FILE"
else
    echo "$(date) Rendering already done, skipping..."
fi

echo "----------------------------------------------------------------"

# Stage 5: Metrics
if [ ! -d "$OUT_DIR/$SCENE_TYPE/$SCENE_NAME/train/5_metrics/v0_render/albedo" ]; then
    echo "$(date) Evaluating Albedo Metrics..."
    LOG_FILE="$OUT_DIR/$SCENE_TYPE/$SCENE_NAME/train/5_metrics/v0_render/albedo/logs.log"
    mkdir -p "$(dirname "$LOG_FILE")"
    COMMAND="python -m iif.job task=5_metrics/v0_albedo component/scene@task.scene=$SCENE_TYPE/$SCENE_NAME paths.out_name=$OUT_DIR"
    echo "$COMMAND" | tee -a "$LOG_FILE"
    eval "$COMMAND" 2>&1 | tee -a "$LOG_FILE"
else
    echo "$(date) Albedo Metrics already evaluated, skipping..."
fi

echo "----------------------------------------------------------------"

if [ ! -d "$OUT_DIR/$SCENE_TYPE/$SCENE_NAME/train/5_metrics/v0_render/albedo_scaled" ]; then
    echo "$(date) Evaluating Albedo Scaled Metrics..."
    LOG_FILE="$OUT_DIR/$SCENE_TYPE/$SCENE_NAME/train/5_metrics/v0_render/albedo_scaled/logs.log"
    mkdir -p "$(dirname "$LOG_FILE")"
    COMMAND="python -m iif.job task=5_metrics/v0_albedo_scaled component/scene@task.scene=$SCENE_TYPE/$SCENE_NAME paths.out_name=$OUT_DIR"
    echo "$COMMAND" | tee -a "$LOG_FILE"
    eval "$COMMAND" 2>&1 | tee -a "$LOG_FILE"
else
    echo "$(date) Albedo Scaled Metrics already evaluated, skipping..."
fi

echo "----------------------------------------------------------------"
  
if [ ! -d "$OUT_DIR/$SCENE_TYPE/$SCENE_NAME/train/5_metrics/v0_render/roughness" ]; then
    echo "$(date) Evaluating Roughness Metrics..."
    LOG_FILE="$OUT_DIR/$SCENE_TYPE/$SCENE_NAME/train/5_metrics/v0_render/roughness/logs.log"
    mkdir -p "$(dirname "$LOG_FILE")"
    COMMAND="python -m iif.job task=5_metrics/v0_roughness component/scene@task.scene=$SCENE_TYPE/$SCENE_NAME paths.out_name=$OUT_DIR"
    echo "$COMMAND" | tee -a "$LOG_FILE"
    eval "$COMMAND" 2>&1 | tee -a "$LOG_FILE"
else
    echo "$(date) Roughness Metrics already evaluated, skipping..."
fi

echo "----------------------------------------------------------------"

if [ ! -d "$OUT_DIR/$SCENE_TYPE/$SCENE_NAME/train/5_metrics/v0_render/metallic" ]; then
    echo "$(date) Evaluating Metallic Metrics..."
    LOG_FILE="$OUT_DIR/$SCENE_TYPE/$SCENE_NAME/train/5_metrics/v0_render/metallic/logs.log"
    mkdir -p "$(dirname "$LOG_FILE")"
    COMMAND="python -m iif.job task=5_metrics/v0_metallic component/scene@task.scene=$SCENE_TYPE/$SCENE_NAME paths.out_name=$OUT_DIR"
    echo "$COMMAND" | tee -a "$LOG_FILE"
    eval "$COMMAND" 2>&1 | tee -a "$LOG_FILE"
else
    echo "$(date) Metallic Metrics already evaluated, skipping..."
fi

echo "----------------------------------------------------------------"

if [ ! -d "$OUT_DIR/$SCENE_TYPE/$SCENE_NAME/train/5_metrics/v0_render/emission" ]; then
    echo "$(date) Evaluating Emission Metrics..."
    LOG_FILE="$OUT_DIR/$SCENE_TYPE/$SCENE_NAME/train/5_metrics/v0_render/emission/logs.log"
    mkdir -p "$(dirname "$LOG_FILE")"
    COMMAND="python -m iif.job task=5_metrics/v0_emission component/scene@task.scene=$SCENE_TYPE/$SCENE_NAME paths.out_name=$OUT_DIR"
    echo "$COMMAND" | tee -a "$LOG_FILE"
    eval "$COMMAND" 2>&1 | tee -a "$LOG_FILE"
else
    echo "$(date) Emission Metrics already evaluated, skipping..."
fi

echo "----------------------------------------------------------------"

if [ ! -d "$OUT_DIR/$SCENE_TYPE/$SCENE_NAME/train/5_metrics/v0_render/rgbs_hdr" ]; then
    echo "$(date) Evaluating RGBs HDR Metrics..."
    LOG_FILE="$OUT_DIR/$SCENE_TYPE/$SCENE_NAME/train/5_metrics/v0_render/rgbs_hdr/logs.log"
    mkdir -p "$(dirname "$LOG_FILE")"
    COMMAND="python -m iif.job task=5_metrics/v0_rgbs_hdr component/scene@task.scene=$SCENE_TYPE/$SCENE_NAME paths.out_name=$OUT_DIR"
    echo "$COMMAND" | tee -a "$LOG_FILE"
    eval "$COMMAND" 2>&1 | tee -a "$LOG_FILE"
else
    echo "$(date) RGBs HDR Metrics already evaluated, skipping..."
fi

echo "----------------------------------------------------------------"

if [ ! -d "$OUT_DIR/$SCENE_TYPE/$SCENE_NAME/train/5_metrics/v0_render/rgbs_ldr" ]; then
    echo "$(date) Evaluating RGBs LDR Metrics..."
    LOG_FILE="$OUT_DIR/$SCENE_TYPE/$SCENE_NAME/train/5_metrics/v0_render/rgbs_ldr/logs.log"
    mkdir -p "$(dirname "$LOG_FILE")"
    COMMAND="python -m iif.job task=5_metrics/v0_rgbs_ldr component/scene@task.scene=$SCENE_TYPE/$SCENE_NAME paths.out_name=$OUT_DIR"
    echo "$COMMAND" | tee -a "$LOG_FILE"
    eval "$COMMAND" 2>&1 | tee -a "$LOG_FILE"
else
    echo "$(date) RGBs LDR Metrics already evaluated, skipping..."
fi

echo "----------------------------------------------------------------"