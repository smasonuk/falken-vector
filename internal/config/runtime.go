package config

import (
	"io"
	"time"
)

const DefaultTimeout = 120 * time.Second

type Runtime struct {
	Paths   Paths
	Timeout time.Duration
	Verbose bool
	Out     io.Writer
	ErrOut  io.Writer
}
