{
  inputs = {
    nixpkgs.url = "github:cachix/devenv-nixpkgs/rolling";
    systems.url = "github:nix-systems/default";
    devenv.url = "github:cachix/devenv";
    devenv.inputs.nixpkgs.follows = "nixpkgs";
    nixpkgs-python.url = "github:cachix/nixpkgs-python";                                                                                                                                           
    nixpkgs-python.inputs = { nixpkgs.follows = "nixpkgs"; };
  };

  nixConfig = {
    extra-trusted-public-keys = "devenv.cachix.org-1:w1cLUi8dv3hnoSPGAuibQv+f9TZLr6cv/Hm9XgU50cw=";
    extra-substituters = "https://devenv.cachix.org";
  };

  outputs = { self, nixpkgs, devenv, systems, ... } @ inputs:
    let
      forEachSystem = nixpkgs.lib.genAttrs (import systems);
    in
    {
      packages = forEachSystem (system: {
        devenv-up = self.devShells.${system}.default.config.procfileScript;
      });

      devShells = forEachSystem
        (system:
          let
            # pkgs = nixpkgs.legacyPackages.${system};
            pkgs = import inputs.nixpkgs {
              config.allowUnfree = true;
              config.cudaSupport = true;
            };
          in
          {
            default = devenv.lib.mkShell {
              inherit inputs pkgs;
              modules = [
                {
                  packages = with pkgs; [
                    cudatoolkit
                    portaudio
                    (python3.withPackages (python-pkgs: with python-pkgs; [
                      jupyter-all
                      jupyter-server
                      ipykernel
                      ipywidgets

                      pyaudio
                      openai-whisper
                      sounddevice
                      rich
                      pyttsx3

                      torch
                      torchaudio
                      torchvision
                      pandas
                      numpy
                      nltk
                    ]))
                  ];

              	enterShell = ''
              	  python -c "import torch; print(f'Cuda Enabled: {torch.cuda.is_available()}')"
              	'';

              	dotenv.enable = true;

              	languages.python = {
                  enable = true;
                  # version = "3.11.10";
                  # version = "3.11.10";
                  # poetry.enable = true;
                  uv.enable = true;
                  venv.enable = true;
                  venv.requirements = ''
                    langchain
                    langchain-core
                    langchain-text-splitters
                    langchainplus-sdk
                    langchain-openai
                    langchain-ollama
                    langchain-cli
                    langchain-community
                    langgraph
                    langgraph-checkpoint
                    langgraph-checkpoint-postgres
                    langgraph-checkpoint-sqlite
                  '';
                };
              }
            ];
          };
        }
      );
    };
}
