#!/usr/bin/env bash
set -euo pipefail
# Bootstrap as container root, then keep ordinary research files owned by the
# laptop user. The researcher can sudo any command inside this container.
lab_uid="${LAB_UID:?LAB_UID is required}"
lab_gid="${LAB_GID:?LAB_GID is required}"
if [[ ! "$lab_uid" =~ ^[1-9][0-9]*$ || ! "$lab_gid" =~ ^[0-9]+$ ]]; then
  echo 'LAB_UID must be a non-root numeric UID and LAB_GID a numeric GID.' >&2
  exit 2
fi
if ! getent group "$lab_gid" >/dev/null; then
  groupadd --gid "$lab_gid" "lab-group-$lab_gid"
fi
if ! getent passwd "$lab_uid" >/dev/null; then
  useradd --uid "$lab_uid" --gid "$lab_gid" --home-dir /home/research --no-create-home "lab-user-$lab_uid"
fi
install -d -o "$lab_uid" -g "$lab_gid" -m 700 /home/research
printf '#%s ALL=(ALL:ALL) NOPASSWD: ALL\n' "$lab_uid" > /etc/sudoers.d/research
chmod 440 /etc/sudoers.d/research
visudo --check --file=/etc/sudoers.d/research >/dev/null
umask 0022
export HOME=/home/research
exec setpriv --reuid "$lab_uid" --regid "$lab_gid" --init-groups "$@"
