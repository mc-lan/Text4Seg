import os


def test_eval_llm():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    from swift.llm import eval_main, EvalArguments
    eval_main(EvalArguments(model_type='qwen1half-7b-chat', eval_dataset='ARC_c', infer_backend='lmdeploy'))


def test_eval_vlm():
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    from swift.llm import eval_main, EvalArguments
    eval_main(EvalArguments(model_type='internvl2-4b', eval_dataset='RealWorldQA', infer_backend='lmdeploy'))


def test_pt():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    from swift.llm import pt_main, PtArguments
    pt_main(PtArguments(model_type='qwen-1_8b-chat', dataset='alpaca-zh', sft_type='lora', tuner_backend='swift'))


def test_vlm_sft():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    from swift.llm import sft_main, SftArguments, infer_main, InferArguments
    output = sft_main(SftArguments(model_type='idefics3-8b-llama3', dataset='coco-en-mini#100'))
    last_model_checkpoint = output['last_model_checkpoint']
    infer_main(InferArguments(ckpt_dir=last_model_checkpoint, load_dataset_config=True, merge_lora=True))


if __name__ == '__main__':
    # test_eval_llm()
    # test_eval_vlm()
    # test_pt()
    test_vlm_sft()
