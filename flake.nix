{
  description = "PostgreSQL 17 with pgvector, llama.cpp with CUDA";
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

  outputs = { self, nixpkgs }:
    let
      forAllSystems = nixpkgs.lib.genAttrs [
        "aarch64-darwin"
        "x86_64-darwin"
        "x86_64-linux"
        "aarch64-linux"
      ];
    in {
      packages = forAllSystems (system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          pkgsCuda = import nixpkgs {
            inherit system;
            config = { allowUnfree = true; cudaSupport = true; };
          };
        in {
          pgvector = pkgs.postgresql_17.withPackages (p: [ p.pgvector ]);
          llama-cpp = if system == "x86_64-linux"
            then pkgsCuda.llama-cpp
            else pkgs.llama-cpp;
        }
      );
    };
}
