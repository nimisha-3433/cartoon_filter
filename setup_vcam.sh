#!/usr/bin/env bash
# Creates a virtual webcam at /dev/video<N> (default /dev/video10) with v4l2loopback.
#   ./setup_vcam.sh            -> /dev/video10 labelled "CartoonCam"
#   ./setup_vcam.sh 12 "MyCam" -> /dev/video12
#   ./setup_vcam.sh --persist  -> also recreate it automatically at every boot
set -euo pipefail

PERSIST=0
if [[ "${1:-}" == "--persist" ]]; then PERSIST=1; shift; fi
NR="${1:-10}"
LABEL="${2:-CartoonCam}"

if ! modinfo v4l2loopback >/dev/null 2>&1; then
    echo "v4l2loopback is not installed:  sudo apt install v4l2loopback-dkms v4l2loopback-utils"
    exit 1
fi

# exclusive_caps=1 makes the device advertise itself as a *capture* device once a
# producer is writing - Chrome/Chromium (Meet, WhatsApp Web) and Zoom need this.
# max_buffers=2 keeps the queue short so the virtual camera adds little latency.
OPTS="devices=1 video_nr=${NR} card_label=${LABEL} exclusive_caps=1 max_buffers=2"

sudo modprobe -r v4l2loopback 2>/dev/null || {
    echo "Could not unload v4l2loopback - close any app using the virtual camera and retry."; }
if ! sudo modprobe v4l2loopback ${OPTS}; then
    echo
    echo "modprobe failed. If the message was 'Key was rejected by service', Secure Boot is"
    echo "blocking the unsigned DKMS module. Either disable Secure Boot in the BIOS, or enrol"
    echo "the DKMS key:  sudo mokutil --import /var/lib/dkms/mok.pub   and reboot."
    exit 1
fi
sleep 0.5

if [[ ! -e "/dev/video${NR}" ]]; then
    echo "/dev/video${NR} did not appear"; exit 1
fi
echo "Virtual camera ready: /dev/video${NR} (\"${LABEL}\")"
v4l2-ctl --list-devices 2>/dev/null || true

if [[ $PERSIST -eq 1 ]]; then
    echo "v4l2loopback" | sudo tee /etc/modules-load.d/v4l2loopback.conf >/dev/null
    echo "options v4l2loopback ${OPTS}" | sudo tee /etc/modprobe.d/v4l2loopback.conf >/dev/null
    echo "Persisted: will be created at every boot."
fi

cat << MSG

Next steps
  1. Run the filter:          ./cartoon /dev/video${NR}     (or ./run.sh meet)
  2. In Zoom / Meet / WhatsApp Web choose the camera "${LABEL}".
     Start the filter FIRST, then open the call app (with exclusive_caps=1 the
     device only shows up as a camera while something is writing to it).
MSG
