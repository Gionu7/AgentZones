Put corporate root CA certificates here as `*.crt` (PEM) before `docker build`.
They are added to the image trust store and used by pip, requests, curl and git.
This folder is copied as-is; `.crt` files are git-ignored so CAs are not committed.
