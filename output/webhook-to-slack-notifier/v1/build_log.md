# Build Log: Webhook to Slack Notifier

**Slug:** webhook-to-slack-notifier
**Version:** 1
**Built:** 2026-02-16T21:27:47Z

## Original Request

> When a webhook fires, parse the data and notify Slack

## Scenario Structure

- **Trigger:** Incoming Webhook (gateway:CustomWebHook)
- **Modules:** 2
- **Connections:** 2

### Modules

- [2] Parse Webhook Body (json:ParseJSON)
- [3] Post to Slack (slack:PostMessage)

## Validation

- **Checks Run:** 48
- **Checks Passed:** 48
- **Errors:** 0
- **Warnings:** 0

## Confidence

- **Score:** 0.98
- **Grade:** A
- **Score affected by: 1 assumption(s).**

## Agent Notes

- Assumed webhook payload contains event_type and source fields
