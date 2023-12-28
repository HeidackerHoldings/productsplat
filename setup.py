if __name__ == "__main__":
    import torch

    print(f"TORCH-CUDA IS AVAILABLE: {torch.cuda.is_available()}")

    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks

    import os

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import tensorflow as tf

    remove_bg = pipeline(
        Tasks.universal_matting, model="damo/cv_unet_universal-matting"
    )

    print(tf.test.is_built_with_cuda())
    print(tf.test.is_gpu_available())
    print(tf.test.is_built_with_gpu_support())
