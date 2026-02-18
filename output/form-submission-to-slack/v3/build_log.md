# Build Log: Form Submission to Slack

**Slug:** form-submission-to-slack
**Version:** 3
**Built:** 2026-02-17T04:25:02Z

## Original Request

> When someone submits a form, parse the data and notify the team on Slack

## Scenario Structure

- **Trigger:** Receive Form Submission (gateway:CustomWebHook)
- **Modules:** 2
- **Connections:** 2

### Modules

- [2] Parse Form Data (json:ParseJSON)
- [3] Notify Team on Slack (slack:PostMessage)

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

- Assumed webhook payload contains 'name' and 'email' fields
- Slack channel #form-submissions must exist
