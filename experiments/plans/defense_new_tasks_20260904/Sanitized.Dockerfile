# Derive a small whiteout layer. Original image layers and dependencies remain.
ARG SOURCE_IMAGE
FROM ${SOURCE_IMAGE}
ARG SOURCE_ID
ARG SOURCE_DIGEST
ARG RECIPE_SHA256
ARG AGENT_UID=501
ARG AGENT_GID=20
USER root
WORKDIR /
RUN set -eu; \
    rm -rf /app; test ! -e /app; test ! -L /app; \
    if ! getent group "${AGENT_GID}" >/dev/null; then \
        if getent group "repoeval${AGENT_GID}" >/dev/null; then exit 1; fi; \
        printf 'repoeval%s:x:%s:\n' "${AGENT_GID}" "${AGENT_GID}" >> /etc/group; \
    fi; \
    if ! getent passwd "${AGENT_UID}" >/dev/null; then \
        if getent passwd "repoeval${AGENT_UID}" >/dev/null; then exit 1; fi; \
        printf 'repoeval%s:x:%s:%s:Benchmark agent:/home/agent:/bin/sh\n' "${AGENT_UID}" "${AGENT_UID}" "${AGENT_GID}" >> /etc/passwd; \
    fi
ENV PYTHONPATH=/workspace/lib:/workspace/test/lib:/workspace
WORKDIR /workspace
LABEL repo-eval.sanitizer="remove-baked-app-v1" \
      repo-eval.source-id="${SOURCE_ID}" \
      repo-eval.source-digest="${SOURCE_DIGEST}" \
      repo-eval.recipe-sha256="${RECIPE_SHA256}"
