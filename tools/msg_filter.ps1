# PowerShell message filter for git filter-branch --msg-filter
# Reads the commit message from stdin and outputs the transformed message

$m = [Console]::In.ReadToEnd()

$map = @{
  'Update documentation and configuration for clarity and environment settings'              = 'docs: Refine documentation and configuration for public release';
  'Update demo_local.py'                                                                     = 'docs: Add script for local pipeline demonstration';
  'Complete UI implementation with functional 3D viewer placeholders and end-to-end testing' = 'feat: Implement interactive web UI with 3D model viewer';
  'Add .env file, directory structure, UI templates and basic FastAPI enhancements'          = 'build: Establish foundational project structure and dependencies';
  'feat: update .gitignore to include specific GitHub instructions directory'                = 'chore: update .gitignore to include GitHub instructions directory';
  'Initial plan'                                                                             = 'docs: Add initial implementation plan';
  'feat: Initial commit of im2fit prototype with FastAPI and Azure integration'              = 'feat: Initialize core pipeline with FastAPI and Azure integrations';
}

foreach ($k in $map.Keys) {
  if ($m.StartsWith($k)) {
    $m = $map[$k] + "`n`nIm2fit history cleanup."
    break
  }
}

[Console]::Out.Write($m)
