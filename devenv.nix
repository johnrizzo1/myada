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
    with pkgs-stable; [
      git
      cudatoolkit
      portaudio
      (python3.withPackages (python-pkgs: with python-pkgs; [
        accelerate
        bitsandbytes
        datasets
        diffusers
        ftfy
        graphviz
        ipykernel
        ipympl
        ipywidgets
        jupyter-all
        jupyter-server
        jupyterlab
        matplotlib
        numpy
        openai-whisper
        pandas
        peft
        pip
        pyaudio
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
        torch
        torchaudio
        torchvision
        xformers
      ]))
      pkgs-unstable.python311Packages.huggingface-hub
      pkgs-unstable.python311Packages.langchain-community
      pkgs-unstable.python311Packages.langchain-core
      pkgs-unstable.python311Packages.langchain-ollama
      pkgs-unstable.python311Packages.langchain-openai
      pkgs-unstable.python311Packages.langchain-text-splitters
      pkgs-unstable.python311Packages.langchainplus-sdk
      pkgs-unstable.python311Packages.langgraph
      pkgs-unstable.python311Packages.langgraph-checkpoint
      pkgs-unstable.python311Packages.langgraph-checkpoint-postgres
      pkgs-unstable.python311Packages.langgraph-checkpoint-sqlite
      pkgs-unstable.python311Packages.llama-index
      pkgs-unstable.python311Packages.llama-index-agent-openai
      pkgs-unstable.python311Packages.nltk
    ];
  
  languages.nix.enable = true;
  languages.python = {
    enable = true;
    # uv.enable = true;
    venv.enable = true;
    venv.requirements = ''
      bark
      langchain-cli
      llama-index-tools-google
      llama-index-tools-wikipedia
    '';
  };

  dotenv.enable = true;
  difftastic.enable = true;
  starship.enable = true;

  enterShell = ''
    git --version
    echo -n "Cuda Enabled: "; python -c "import torch; print(torch.cuda.is_available())"
    echo -n "llama google tools: "; python -c "import llama_index.tools.google"; [[ $? = 0 ]] && echo 'Imported' || echo 'Failed to import'
    echo -n "Bark: "; python -c "import bark"; [[ $? = 0 ]] && echo 'Imported' || echo 'Failed to import'
  '';

  # See full reference at https://devenv.sh/reference/options/
}
