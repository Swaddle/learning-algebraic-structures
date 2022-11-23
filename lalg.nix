{ pkgs ? import <nixpkgs> {} }:


pkgs.mkShell {
  buildInputs = [
    pkgs.gap
    pkgs.python310
    pkgs.python310Packages.pytorchWithoutCuda
    pkgs.python310Packages.pip
    pkgs.python310Packages.virtualenv
    pkgs.sage
    pkgs.git 
    pkgs.git-lfs
  ];
}
