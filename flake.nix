{
  description = "ADA Another Digital Assistant";

  inputs = {
    dream2nix.url = "github:nix-community/dream2nix";
    nixpkgs.follows = "dream2nix/nixpkgs";
  };

  outputs = {
    self,
    dream2nix,
    nixpkgs,
  }: let
    eachSystem = nixpkgs.lib.genAttrs [
      "aarch64-darwin"
      "aarch64-linux"
      "x86_64-darwin"
      "x86_64-linux"
    ];
  in {
    packages = eachSystem (system: {
      default = dream2nix.lib.evalModules {
        packageSets.nixpkgs = nixpkgs.legacyPackages.${system};
        modules = [
          ./default.nix
          {
            paths.projectRoot = ./.;
            paths.projectRootFile = "flake.nix";
            paths.package = ./.;
          }
        ];
      };
    });
    devShells = eachSystem (system: let
        pkgs = nixpkgs.legacyPackages.${system};
        ada = self.packages.${system}.default;
      in {
        default = pkgs.mkShell {
          inputsFrom = [ada.devShell];
          packages = with pkgs; [
            portaudio
            ada

            
          ];
        };
      });
  };
}
