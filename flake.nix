{
  description = "Another Digital Assistant";
  
  inputs = {
    flake-schemas.url = "https://flakehub.com/f/DeterminateSystems/flake-schemas/*";
    nixpkgs.url = "https://flakehub.com/f/NixOS/nixpkgs/*";
  };
  
  outputs = { self, flake-schemas, nixpkgs }:
    let
      supportedSystems = [ "aarch64-darwin" "x86_64-linux" ];
      forEachSupportedSystem = f: nixpkgs.lib.genAttrs supportedSystems (system: f {
        pkgs = import nixpkgs { 
          inherit system;
          config.allowUnfree = true;
          config.cudaSupport = true;
        };
      });
    in {
      schemas = flake-schemas.schemas;
      
      devShells = forEachSupportedSystem ({ pkgs }: {
        default = pkgs.mkShell {
          packages = with pkgs; [
            curl
            git
            jq
            wget
            nixpkgs-fmt
            uv
            ffmpeg
            portaudio
            # cudatoolkit
            (python3.withPackages (python-pkgs: with python-pkgs; [
              torch
              torchvision
              python-dotenv
              pip
              ipykernel
              jupyterlab
              transformers
              pyaudio
              sounddevice
            ]))
          ];
          
          env = {
            BOO = "yah";
          };
        };
      });
    };
}
