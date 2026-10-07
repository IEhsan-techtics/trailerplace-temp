"""The follow-up agent: nudges Messenger customers who went quiet after our message.

Run hourly by the Azure Container Apps Job ``trailerplace-followup``:

    python -m src.followup

It finds conversations where our last message has gone unanswered for FOLLOWUP_FIRST_AFTER_HOURS
(then FOLLOWUP_SECOND_AFTER_HOURS), lets the model decide whether a follow-up is worth sending
and write it, sends it on Messenger and records it in the conversation. Off unless
FOLLOWUP_ENABLED is on - only the Azure job sets it.
"""
