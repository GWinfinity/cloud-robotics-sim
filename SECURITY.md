# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 2.0.x   | :white_check_mark: |
| < 2.0   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability in Cloud Robotics Simulation Platform, please report it to us as soon as possible.

### How to Report

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report them via email to:

📧 **security@cloudrobotics.dev**

Please include:
- A description of the vulnerability
- Steps to reproduce the issue
- Possible impact of the vulnerability
- Any suggested fixes (if available)

### Response Timeline

- **Acknowledgment**: Within 48 hours
- **Initial assessment**: Within 5 business days
- **Fix released**: Depends on severity (critical: 7 days, high: 30 days, medium/low: next release)

## Security Best Practices

When using this software:

1. **Keep dependencies updated** - Regularly update Genesis and other dependencies
2. **Use isolated environments** - Run simulations in containers or VMs when possible
3. **Validate inputs** - Always validate configuration files and user inputs
4. **Secure deployment** - Follow Kubernetes security best practices for cloud deployment

## Known Security Considerations

- Genesis physics engine runs with full GPU access
- Cloud deployment requires careful network configuration
- Simulation environments should not be exposed to untrusted networks without authentication
