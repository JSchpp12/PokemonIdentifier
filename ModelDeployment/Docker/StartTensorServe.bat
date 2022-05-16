docker run -p 8501:8501 `
    --name test `
    -v ${PWD}'/../Models:/models/' `
    -e 'MODEL_NAME=classifier' `
    --gpus all -t tensorflow/serving:latest-gpu