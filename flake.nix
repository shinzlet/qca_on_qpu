{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs, ... }: {
    devShells.x86_64-darwin.default = nixpkgs.legacyPackages.x86_64-darwin.mkShell {
      buildInputs = [
        # nixpkgs.legacyPackages.x86_64-darwin.python39
        # Other dependencies
        (nixpkgs.legacyPackages.x86_64-darwin.python39.withPackages (ps: [
          # ps.numpy
          # ps.matplotlib
        ]))
      ];
      shellHook = ''
        # Create virtual environment if it doesn't exist
        if [ ! -d ".venv" ]; then
          python -m venv .venv
        fi

        # Activate virtual environment
        source .venv/bin/activate

        # Install dwave-ocean-sdk if it's not already installed
        if ! python -c "import dwave" > /dev/null 2>&1; then
          pip install numpy
          pip install matplotlib
          pip install scipy
          pip install dwave-ocean-sdk
        fi
      '';
    };
  };
}

