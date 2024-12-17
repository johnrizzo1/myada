{
  description = "Another Digital Assistant";
  
  inputs = {
    flake-schemas.url = "https://flakehub.com/f/DeterminateSystems/flake-schemas/*";

    nixpkgs.url = "https://flakehub.com/f/NixOS/nixpkgs/*";
  };
  
  outputs = { self, flake-schemas, nixpkgs }:
    let
      supportedSystems = [ "aarch64-darwin" ];
      forEachSupportedSystem = f: nixpkgs.lib.genAttrs supportedSystems (system: f {
        pkgs = import nixpkgs { inherit system; };
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
            ffmpeg # for whisper
            # (python3.withPackages (python-pkgs: with python-pkgs; [
            #   pip
            #   openai-whisper
            # ]))
          ];
          
          env = {
            BOO = "yah";
          };
        };
      });
    };
}
