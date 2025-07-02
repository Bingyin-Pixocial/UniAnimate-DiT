#!/bin/bash

# UniAnimate-DiT Docker Runner Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Check if nvidia-docker is available (for GPU support)
if ! command -v nvidia-docker &> /dev/null; then
    print_warning "nvidia-docker not found. GPU support may not work properly."
    print_warning "Please install nvidia-docker for GPU acceleration."
fi

# Function to build the Docker image
build_image() {
    print_status "Building UniAnimate-DiT Docker image..."
    docker-compose build
    print_success "Docker image built successfully!"
}

# Function to run the container
run_container() {
    print_status "Starting UniAnimate-DiT container..."
    docker-compose up -d
    print_success "Container started successfully!"
    print_status "You can now access the container with: docker exec -it unianimate-dit bash"
}

# Function to stop the container
stop_container() {
    print_status "Stopping UniAnimate-DiT container..."
    docker-compose down
    print_success "Container stopped successfully!"
}

# Function to show logs
show_logs() {
    print_status "Showing container logs..."
    docker-compose logs -f
}

# Function to execute a command in the container
exec_command() {
    if [ -z "$1" ]; then
        print_error "No command provided. Usage: $0 exec <command>"
        exit 1
    fi
    print_status "Executing command in container: $1"
    docker exec -it unianimate-dit bash -c "$1"
}

# Function to enter the container shell
enter_shell() {
    print_status "Entering container shell..."
    docker exec -it unianimate-dit bash
}

# Function to show help
show_help() {
    echo "UniAnimate-DiT Docker Runner Script"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  build     Build the Docker image"
    echo "  run       Start the container"
    echo "  stop      Stop the container"
    echo "  restart   Restart the container"
    echo "  logs      Show container logs"
    echo "  shell     Enter the container shell"
    echo "  exec      Execute a command in the container"
    echo "  help      Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 build                    # Build the image"
    echo "  $0 run                      # Start the container"
    echo "  $0 shell                    # Enter container shell"
    echo "  $0 exec 'python --version'  # Check Python version in container"
    echo ""
}

# Main script logic
case "${1:-help}" in
    build)
        build_image
        ;;
    run)
        run_container
        ;;
    stop)
        stop_container
        ;;
    restart)
        stop_container
        run_container
        ;;
    logs)
        show_logs
        ;;
    shell)
        enter_shell
        ;;
    exec)
        exec_command "${@:2}"
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        echo ""
        show_help
        exit 1
        ;;
esac 