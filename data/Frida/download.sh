#!/bin/bash

wget https://frida.fooddata.dk/download

unzip download

rm download

# As for 12/06/2022, Frida does not provide chemical identifiers. We have
#   requested the latest version of the database, but it is not publicly
#   available at the moment.
