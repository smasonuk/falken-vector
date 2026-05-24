test:
	CGO_ENABLED=0 go test ./...

build:
	CGO_ENABLED=0 go build -o bin/falkengo ./cmd/falkengo

lint:
	go vet ./...
