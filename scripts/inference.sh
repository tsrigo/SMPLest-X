#!/bin/bash
CKPT_NAME=$1
FILE_NAME=$2
FPS=${3:-30}
SHOW_SKELETON=${4:-false}

NAME="${FILE_NAME%.*}"
EXT="${FILE_NAME##*.}"

IMG_PATH=./demo/input_frames/$NAME
OUTPUT_PATH=./demo/output_frames/$NAME

mkdir -p $IMG_PATH
mkdir -p $OUTPUT_PATH

# convert video to frames
case "$EXT" in
    mp4|avi|mov|mkv|flv|wmv|webm|mpeg|mpg)
        ffmpeg -i ./demo/$FILE_NAME -f image2 -vf fps=${FPS}/1 -qscale 0 ${IMG_PATH}/%06d.jpg 
        ;;
    jpg|jpeg|png|bmp|gif|tiff|tif|webp|svg)
        cp ./demo/$FILE_NAME $IMG_PATH/000001.$EXT
        ;;
    *)
        echo "Unknown file type."
        exit 1
        ;;
esac

END_COUNT=$(find "$IMG_PATH" -type f | wc -l)

# Build skeleton flag
SKELETON_FLAG=""
if [ "$SHOW_SKELETON" = "true" ] || [ "$SHOW_SKELETON" = "1" ] || [ "$SHOW_SKELETON" = "skeleton" ]; then
    SKELETON_FLAG="--show_skeleton"
fi

# inference with smplest_x
PYTHONPATH=../:$PYTHONPATH \
PYOPENGL_PLATFORM="" \
python main/inference.py \
    --num_gpus 1 \
    --file_name $NAME \
    --ckpt_name $CKPT_NAME \
    --end $END_COUNT \
    $SKELETON_FLAG


# convert frames to video
case "$EXT" in
    mp4|avi|mov|mkv|flv|wmv|webm|mpeg|mpg)
        ffmpeg -y -f image2 -r ${FPS} -i ${OUTPUT_PATH}/%06d.jpg \
        -c:v libopenh264 \
        -vf "format=yuv420p,scale=trunc(iw/2)*2:trunc(ih/2)*2" \
        -allow_skip_frames 1 \
        ./demo/result_${NAME}.mp4
        ;;
    jpg|jpeg|png|bmp|gif|tiff|tif|webp|svg)
        cp $OUTPUT_PATH/000001.$EXT ./demo/result_$FILE_NAME
        ;;
    *)
        exit 1
        ;;
esac

rm -rf ./demo/input_frames
rm -rf ./demo/output_frames

