"""Proxy Doctor: diagnoses and fixes corporate proxy / TLS-interception issues locally.

Pieces:
  catalog      failure classes, fix plans, explanations
  diagnostics  real checks on this machine (+ a simulator for demos/tests)
  features     observation -> feature vector
  synth        synthetic failure cases (training data, per-site skew)
  student      small softmax classifier, distillable + federated (FL task)
  mcp_server   MCP server exposing the diagnostic/fix tools
  agent        the Agent Zone agent that ties it together; consulting and sharing
               fixes with the other zones is the generic agentzones.peers.PeerAgent
"""
