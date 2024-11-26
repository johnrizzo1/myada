{ pkgs, lib, config, inputs, ... }:

{
  packages =
    let
      pkgs-stable = import inputs.nixpkgs {
        inherit (pkgs.stdenv) system;

        config.allowUnfree = true;
        config.cudaSupport = true;
      };

      pkgs-unstable = import inputs.nixpkgs-unstable {
        inherit (pkgs.stdenv) system;

        config.allowUnfree = true;
        config.cudaSupport = true;
      };
    in
    with pkgs-unstable; [
      git
      cudatoolkit
      portaudio
      nixpkgs-fmt
    ];
  
  languages.nix.enable = true;
  languages.python = {
    enable = true;
    venv.enable = true;
    venv.requirements = ''
      accelerate
      bark
      bitsandbytes
      datasets
      diffusers
      docarray
      ftfy
      graphviz
      huggingface-hub
      ipykernel
      ipympl
      ipywidgets
      jupyter[all]
      jupyterlab
      langchain
      langchain-community
      langchain-core
      langchain-ollama
      langchain-openai
      langchain-anthropic
      langchain-text-splitters
      langchainplus-sdk
      langgraph
      langgraph-checkpoint
      langgraph-checkpoint-postgres
      langgraph-checkpoint-sqlite
      llama-index-agent-openai
      llama-index-tools-google
      llama-index-tools-wikipedia
      matplotlib
      nltk
      numpy
      openai-whisper
      pandas
      peft
      pip
      pyaudio
      pyowm
      python-dotenv
      pyttsx3
      requests
      rich
      scipy
      seaborn
      sounddevice
      sqlalchemy
      statsmodels
      tensorboard
      tiktoken
      torch
      torchaudio
      torchvision
      xformers
    '';
  };

  dotenv.enable = true;
  difftastic.enable = true;
  starship.enable = true;

  enterShell = ''
    git --version
    echo -n "Python Version: "; python --version
    echo "Pythons:"
    which -a python
    echo -n "Cuda Enabled: "; python -c "import torch; print(torch.cuda.is_available())"
    echo -n "llama google tools: "; python -c "import llama_index.tools.google"; [[ $? = 0 ]] && echo 'Imported' || echo 'Failed to import'
    echo -n "Bark: "; python -c "import bark"; [[ $? = 0 ]] && echo 'Imported' || echo 'Failed to import'
  '';

  # See full reference at https://devenv.sh/reference/options/
}
