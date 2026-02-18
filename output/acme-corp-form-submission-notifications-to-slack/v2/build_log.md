# Build Log: Acme Corp: Form submission notifications to Slack

**Slug:** acme-corp-form-submission-notifications-to-slack
**Version:** 2
**Built:** 2026-02-18T05:36:26Z

## Original Request

> When a form is submitted, notify the team in Slack

## Scenario Structure

- **Trigger:** Incoming form submission webhook (gateway:CustomWebHook)
- **Modules:** 2
- **Connections:** 2

### Modules

- [2] Parse the incoming JSON payload (json:ParseJSON)
- [3] Post notification to Slack channel (slack:PostMessage)

## Validation

- **Checks Run:** 48
- **Checks Passed:** 48
- **Errors:** 0
- **Warnings:** 0

## Confidence

- **Score:** 0.96
- **Grade:** A
- **Score affected by: 2 assumption(s).**

## Agent Notes

- Auto-resolved 'Parse the incoming JSON payload' → json:ParseJSON
- Auto-resolved 'Post notification to Slack channel' → slack:PostMessage
