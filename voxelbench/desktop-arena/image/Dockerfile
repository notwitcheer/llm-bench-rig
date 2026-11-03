# desktop-arena image: the official desktop sandbox + the everyday GUI apps its Xfce image lacks (no editor, no file manager, no spreadsheet).
# Derived on purpose and labelled in every result; the stock tag stays the control for the screen/driver stack.
FROM nousresearch/hermes-sandbox:desktop
USER root
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends mousepad thunar gnumeric at-spi2-core fonts-dejavu-core && rm -rf /var/lib/apt/lists/*
