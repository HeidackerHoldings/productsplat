ROOT="$(cd -P "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "Building $ROOT/docker/Dockerfile.dev"
docker build -f "$ROOT/docker/Dockerfile.dev" -t productsplatdev $ROOT