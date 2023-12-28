ROOT="$(cd -P "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "Building $ROOT/docker/Dockerfile.prod"
docker build -f "$ROOT/docker/Dockerfile.prod" -t productsplat $ROOT