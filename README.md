# MT Exercise 2: Pytorch RNN Language Models

This repo shows how to train neural language models using [Pytorch example code](https://github.com/pytorch/examples/tree/master/word_language_model). Thanks to Emma van den Bold, the original author of these scripts.

# Parameter tuning: Experimenting with dropout

We trained 5 different language models using various dropout settings (0.0, 0.2, 0.4, 0.6, 0.8) on the novel *Pride and Prejudice* by Jane Austen, sourced from [Project Gutenberg](https://www.gutenberg.org/ebooks/1342). Our goal was to compare perplexities across different dropout values and select the best performing model.

# Requirements

- This only works on a Unix-like system, with bash.
- Python 3 must be installed on your system, i.e. the command `python3` must be available
- Make sure virtualenv is installed on your system. To install, e.g.

    `pip install virtualenv`

# Steps

Clone this repository in the desired place:

    git clone https://github.com/marpng/mt-exercise-02
    cd mt-exercise-02

Create a new virtualenv that uses Python 3. Please make sure to run this command outside of any virtual Python environment:

    ./scripts/make_virtualenv.sh

**Important**: Then activate the env by executing the `source` command that is output by the shell script above.

Download and install required software:

    ./scripts/install_packages.sh

Download and preprocess data:

    ./scripts/download_data.sh

Train a model:
We trained 5 models with different dropout values:

    ./scripts/train.sh
Modify the dropout value in scripts/train.sh and rerun it to train other models. All models are saved under models/, and corresponding perplexity logs under logs/.

Analyze Perplexity & Plot:

```bash
python scripts/plot_logs.py

The resulting plots (train_ppl_curve.png and valid_ppl_curve.png) will be saved in the logs/ directory.

Generate (sample) some text from a trained model with:

    ./scripts/generate.sh
Edit generate.sh to choose which model to load

