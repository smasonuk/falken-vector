//go:build !aix && !darwin && !dragonfly && !freebsd && !linux && !netbsd && !openbsd && !solaris

package config

func processRunning(pid int) bool {
	return pid > 0
}
