# Labelling guide

Version 1.0.0, kept with the decision contract.

Label the queue that must act first.

If any part of the message says the customer did not make a transaction, or that a card or phone is lost, stolen, or compromised, label `fraud`.

If the message fits no queue, or it is not English, mark it `out_of_scope`. Do not force a queue. Those rows are kept for hand-off tests and are not training labels.

If two queues fit equally and the fraud rule does not apply, mark it `ambiguous` and send it to the labelling lead.

| Customer says | Banking77 intent | Queue |
|---|---|---|
| I lost my card, or my card was stolen | `lost_or_stolen_card` | `fraud` |
| My phone was stolen | `lost_or_stolen_phone` | `fraud` |
| I was charged twice | `transaction_charged_twice` | `fraud` |
| There's a payment I don't recognise | `card_payment_not_recognised` | `fraud` |
| There's a cash withdrawal I didn't make | `cash_withdrawal_not_recognised` | `fraud` |
| My card payment was declined | `declined_card_payment` | `transfers` |
| My card doesn't work | `card_not_working` | `cards` |
| The exchange rate on my card payment was wrong | `card_payment_wrong_exchange_rate` | `fees` |
| My top-up failed | `top_up_failed` | `cash` |
| I got the wrong amount of cash from the ATM | `wrong_amount_of_cash_received` | `cash` |
| I forgot my app passcode | `passcode_forgotten` | `account` |
| My refund is not showing | `Refund_not_showing_up` | `transfers` |
| Why do you need to verify my source of funds? | `verify_source_of_funds` | `account` |

The business owner has not yet confirmed the fraud mapping for `lost_or_stolen_card`, `lost_or_stolen_phone`, and `transaction_charged_twice`.
