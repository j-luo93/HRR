# Basic usage to get all fillers
`bash get_all_fillers.sh <model_dir> <ckpt_number>  <vocab_file_path> <output_file_prefix> <model> <data_size> <write_size>`

For instance, to extract all embeddings (9974 words) for the baseline model on ptb, run
`bash get_all_fillers.sh reb-ptb-baseline-sgd/train 23275 data/ptb/vocab.txt test-baseline baseline ptb 9974`

To extract all embeddings for the word HRR on ptb, run
`bash get_all_fillers.sh reb-ptb-wordhrr-sgd/train 62342 data/ptb/vocab.txt test-wordhrr hrr ptb 9974`

