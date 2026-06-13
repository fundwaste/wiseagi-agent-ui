{ pkgs ? import <nixpkgs> { } }:
let
  python = pkgs.python312.withPackages (pp: [
    # pp.websockets
    # pp.aiohttp
    pp.httpx
    pp.fastapi
    pp.uvicorn

    # pp.ruff
    pp.mypy
  ]);
in
pkgs.mkShell {
  packages = [
    python

    pkgs.ruff
    pkgs.rlwrap
    pkgs.websocat
    pkgs.claws
    pkgs.nixpkgs-fmt
  ];
}
