#!/usr/bin/env python3
"""units task registry (hand written, frozen by freeze.py before any model output).

Each task = ONE tool + ONE quantity argument whose unit the tool schema names, + an exact rational expected value in that unit.
Each task is written in 4 FORMS (roles): canonical (the argument's own unit), sub (a smaller unit, or a mixed form for money), larger (a bigger unit),
words (number words or a fraction of a larger unit). The quantity is the only thing that changes between the four prompts of a task.

REAL tools: the schema is taken verbatim from the pinned Hermes checkout (schemas.json, extract_schemas.py records file + line).
SYNTHETIC tools: four small tools written here (charge_card, trim_cable, set_delay, weigh_out); labelled synthetic in every result row and report,
NOT labelled inside the schema the model sees.
Each form carries its own machine-readable (num, unit); freeze.py proves num x unit factor == expected exactly for all 160 forms.
"""
from fractions import Fraction

# unit factors RELATIVE TO THE ARGUMENT'S OWN UNIT
FAC = {
    "seconds": {"ms": Fraction(1, 1000), "s": Fraction(1), "min": Fraction(60), "h": Fraction(3600)},
    "milliseconds": {"us": Fraction(1, 1000), "ms": Fraction(1), "s": Fraction(1000), "min": Fraction(60000)},
    "cents": {"cents": Fraction(1), "dollars": Fraction(100), "kdollars": Fraction(100000)},
    "metres": {"mm": Fraction(1, 1000), "cm": Fraction(1, 100), "m": Fraction(1), "km": Fraction(1000)},
    "grams": {"mg": Fraction(1, 1000), "g": Fraction(1), "kg": Fraction(1000)},
}

# tool -> (real?, argument, JSON type, unit family, extra fixed args scored, schema source)
TOOLS = {
    "terminal": dict(real=True, arg="timeout", argtype="integer", fam="seconds", must={}, verb="set"),
    "process": dict(real=True, arg="timeout", argtype="integer", fam="seconds", must={"action": "wait"}, verb="set"),
    "browser_cdp": dict(real=True, arg="timeout", argtype="number", fam="seconds", must={}, verb="set"),
    "browser_exec": dict(real=True, arg="timeout_s", argtype="integer", fam="seconds", must={}, verb="set"),
    "computer_use": dict(real=True, arg="seconds", argtype="number", fam="seconds", must={"action": "wait"}, verb="set"),
    "charge_card": dict(real=False, arg="amount_cents", argtype="integer", fam="cents", must={}, verb="charged"),
    "trim_cable": dict(real=False, arg="length_m", argtype="number", fam="metres", must={}, verb="cut"),
    "set_delay": dict(real=False, arg="delay_ms", argtype="integer", fam="milliseconds", must={}, verb="set"),
    "weigh_out": dict(real=False, arg="mass_g", argtype="number", fam="grams", must={}, verb="dispensed"),
}

SYNTHETIC_SCHEMAS = {
    "charge_card": {"name": "charge_card", "description": "Charge a stored payment card.",
                    "parameters": {"type": "object", "properties": {
                        "card_label": {"type": "string", "description": "Label of the stored card."},
                        "amount_cents": {"type": "integer", "description": "Amount to charge, in cents.", "minimum": 1}},
                        "required": ["card_label", "amount_cents"]}},
    "trim_cable": {"name": "trim_cable", "description": "Cut a length of cable off a spool.",
                   "parameters": {"type": "object", "properties": {
                       "spool": {"type": "string", "description": "Name of the spool."},
                       "length_m": {"type": "number", "description": "Length to cut, in metres."}},
                       "required": ["spool", "length_m"]}},
    "set_delay": {"name": "set_delay", "description": "Set the delay of a named timer.",
                  "parameters": {"type": "object", "properties": {
                      "name": {"type": "string", "description": "Name of the timer."},
                      "delay_ms": {"type": "integer", "description": "Delay to apply, in milliseconds.", "minimum": 1}},
                      "required": ["name", "delay_ms"]}},
    "weigh_out": {"name": "weigh_out", "description": "Dispense an ingredient from the scale.",
                  "parameters": {"type": "object", "properties": {
                      "item": {"type": "string", "description": "Ingredient to dispense."},
                      "mass_g": {"type": "number", "description": "Mass to dispense, in grams."}},
                      "required": ["item", "mass_g"]}},
}


def F(num, unit, text):
    return {"num": num, "unit": unit, "text": text}


def T(id, tool, template, expected, ask, forms):
    roles = ["canonical", "sub", "larger", "words"]
    return {"id": id, "tool": tool, "template": template, "expected": expected, "ask": ask,
            "forms": [dict(role=r, **f) for r, f in zip(roles, forms)]}


TASKS = [
    # ---- terminal.timeout (seconds, integer; foreground max 600 in the schema text) ----
    T("t01", "terminal", "Run `pytest -x tests/` with a timeout of {Q}.", "90", "a 90-second timeout",
      [F("90", "s", "90 seconds"), F("90000", "ms", "90000 milliseconds"), F("3/2", "min", "1.5 minutes"), F("3/2", "min", "a minute and a half")]),
    T("t02", "terminal", "Run `npm run build` with the timeout set to {Q}.", "240", "a 4-minute timeout",
      [F("240", "s", "240 seconds"), F("240000", "ms", "240000 milliseconds"), F("4", "min", "4 minutes"), F("1/15", "h", "a fifteenth of an hour")]),
    T("t03", "terminal", "Execute `rsync -a data/ backup/` with a time limit of {Q}.", "600", "a 10-minute timeout",
      [F("600", "s", "600 seconds"), F("600000", "ms", "600000 milliseconds"), F("10", "min", "10 minutes"), F("1/6", "h", "a sixth of an hour")]),
    T("t04", "terminal", "Check `curl -s https://example.org/health` with a timeout of {Q}.", "20", "a 20-second timeout",
      [F("20", "s", "20 seconds"), F("20000", "ms", "20000 milliseconds"), F("1/3", "min", "one third of a minute"), F("20", "s", "twenty seconds")]),
    T("t05", "terminal", "Run `make -j4` in the project with a timeout of {Q}.", "300", "a 5-minute timeout",
      [F("300", "s", "300 seconds"), F("300000", "ms", "300000 milliseconds"), F("5", "min", "5 minutes"), F("1/12", "h", "a twelfth of an hour")]),
    T("t06", "terminal", "Run `python train_small.py` with the timeout set to {Q}.", "45", "a 45-second timeout",
      [F("45", "s", "45 seconds"), F("45000", "ms", "45000 milliseconds"), F("3/4", "min", "0.75 minutes"), F("3/4", "min", "three quarters of a minute")]),
    T("t07", "terminal", "Please run `docker compose up --build` with a time limit of {Q}.", "480", "an 8-minute timeout",
      [F("480", "s", "480 seconds"), F("480000", "ms", "480000 milliseconds"), F("8", "min", "8 minutes"), F("2/15", "h", "two fifteenths of an hour")]),
    T("t08", "terminal", "Run `git gc` with a timeout of {Q}.", "150", "a 2.5-minute timeout",
      [F("150", "s", "150 seconds"), F("150000", "ms", "150000 milliseconds"), F("5/2", "min", "2.5 minutes"), F("5/2", "min", "two and a half minutes")]),
    # ---- process.timeout (action wait; seconds, integer) ----
    T("p01", "process", "Wait on proc_4dae56ca81f6 until it exits, with a timeout of {Q}.", "120", "a 2-minute timeout",
      [F("120", "s", "120 seconds"), F("120000", "ms", "120000 milliseconds"), F("2", "min", "2 minutes"), F("1/30", "h", "a thirtieth of an hour")]),
    T("p02", "process", "Block on proc_9b21c07d33ef until it exits, with the timeout set to {Q}.", "360", "a 6-minute timeout",
      [F("360", "s", "360 seconds"), F("360000", "ms", "360000 milliseconds"), F("6", "min", "6 minutes"), F("1/10", "h", "a tenth of an hour")]),
    T("p03", "process", "I started proc_51aa0e7c9d42 earlier; wait on it with a time limit of {Q}.", "75", "a 75-second timeout",
      [F("75", "s", "75 seconds"), F("75000", "ms", "75000 milliseconds"), F("5/4", "min", "1.25 minutes"), F("5/4", "min", "one and a quarter minutes")]),
    T("p04", "process", "Hold on proc_c0de77a1b2f3 until it exits, with a timeout of {Q}.", "540", "a 9-minute timeout",
      [F("540", "s", "540 seconds"), F("540000", "ms", "540000 milliseconds"), F("9", "min", "9 minutes"), F("3/20", "h", "three twentieths of an hour")]),
    # ---- browser_cdp.timeout (seconds, number; max 300) ----
    T("c01", "browser_cdp", "On the open tab call the DevTools method Page.captureScreenshot with a timeout of {Q}.", "10", "a 10-second timeout",
      [F("10", "s", "10 seconds"), F("10000", "ms", "10000 milliseconds"), F("1/6", "min", "one sixth of a minute"), F("10", "s", "ten seconds")]),
    T("c02", "browser_cdp", "Send Runtime.evaluate to the current page with the timeout set to {Q}.", "45", "a 45-second timeout",
      [F("45", "s", "45 seconds"), F("45000", "ms", "45000 milliseconds"), F("3/4", "min", "0.75 minutes"), F("3/4", "min", "three quarters of a minute")]),
    T("c03", "browser_cdp", "Call Page.reload on the active tab with a timeout of {Q}.", "120", "a 2-minute timeout",
      [F("120", "s", "120 seconds"), F("120000", "ms", "120000 milliseconds"), F("2", "min", "2 minutes"), F("1/30", "h", "a thirtieth of an hour")]),
    T("c04", "browser_cdp", "Use the Network.getCookies method on the open tab with a time limit of {Q}.", "5/2", "a 2.5-second timeout",
      [F("5/2", "s", "2.5 seconds"), F("2500", "ms", "2500 milliseconds"), F("1/24", "min", "one twenty-fourth of a minute"), F("5/2", "s", "two and a half seconds")]),
    # ---- browser_exec.timeout_s (seconds, integer; max 1800) ----
    T("e01", "browser_exec", "In the browser run `print(page_info())` with a timeout of {Q}.", "600", "a 10-minute timeout",
      [F("600", "s", "600 seconds"), F("600000", "ms", "600000 milliseconds"), F("10", "min", "10 minutes"), F("1/6", "h", "a sixth of an hour")]),
    T("e02", "browser_exec", "Run this browser snippet `print(js('document.title'))` with the timeout set to {Q}.", "30", "a 30-second timeout",
      [F("30", "s", "30 seconds"), F("30000", "ms", "30000 milliseconds"), F("1/2", "min", "0.5 minutes"), F("30", "s", "thirty seconds")]),
    T("e03", "browser_exec", "In the browser run `print(js('location.href'))` with a time limit of {Q}.", "1200", "a 20-minute timeout",
      [F("1200", "s", "1200 seconds"), F("1200000", "ms", "1200000 milliseconds"), F("20", "min", "20 minutes"), F("1/3", "h", "a third of an hour")]),
    T("e04", "browser_exec", "Run `goto_url('https://example.org')` in the browser with a timeout of {Q}.", "180", "a 3-minute timeout",
      [F("180", "s", "180 seconds"), F("180000", "ms", "180000 milliseconds"), F("3", "min", "3 minutes"), F("1/20", "h", "a twentieth of an hour")]),
    # ---- computer_use.seconds (action wait; seconds, number; max 30) ----
    T("u01", "computer_use", "Pause for {Q} so the page can finish loading.", "5/2", "a 2.5-second pause",
      [F("5/2", "s", "2.5 seconds"), F("2500", "ms", "2500 milliseconds"), F("1/24", "min", "one twenty-fourth of a minute"), F("5/2", "s", "two and a half seconds")]),
    T("u02", "computer_use", "Wait {Q} before the next click.", "12", "a 12-second wait",
      [F("12", "s", "12 seconds"), F("12000", "ms", "12000 milliseconds"), F("1/5", "min", "0.2 minutes"), F("1/5", "min", "a fifth of a minute")]),
    T("u03", "computer_use", "Just wait {Q}.", "1/2", "a half-second wait",
      [F("1/2", "s", "0.5 seconds"), F("500", "ms", "500 milliseconds"), F("1/120", "min", "a hundred-and-twentieth of a minute"), F("1/2", "s", "half a second")]),
    T("u04", "computer_use", "Give the app {Q} to open its window.", "20", "a 20-second wait",
      [F("20", "s", "20 seconds"), F("20000", "ms", "20000 milliseconds"), F("1/3", "min", "one third of a minute"), F("20", "s", "twenty seconds")]),
    # ---- SYNTHETIC charge_card.amount_cents ----
    T("d01", "charge_card", "Charge {Q} to the card labelled 'office-visa'.", "1250", "a $12.50 charge",
      [F("1250", "cents", "1250 cents"), F("25/2", "dollars", "12 dollars and 50 cents"), F("25/2", "dollars", "12.50 dollars"), F("25/2", "dollars", "twelve dollars fifty")]),
    T("d02", "charge_card", "Bill {Q} to the card labelled 'team-amex'.", "4999", "a $49.99 charge",
      [F("4999", "cents", "4999 cents"), F("4999/100", "dollars", "49 dollars and 99 cents"), F("4999/100", "dollars", "49.99 dollars"), F("4999/100", "dollars", "forty-nine dollars ninety-nine")]),
    T("d03", "charge_card", "Take {Q} from the 'travel' card.", "120000", "a $1,200 charge",
      [F("120000", "cents", "120000 cents"), F("1200", "dollars", "1200 dollars"), F("6/5", "kdollars", "1.2 thousand dollars"), F("1200", "dollars", "twelve hundred dollars")]),
    T("d04", "charge_card", "Charge {Q} to the 'lab' card.", "75", "a 75-cent charge",
      [F("75", "cents", "75 cents"), F("3/4", "dollars", "three quarters of a dollar"), F("3/4", "dollars", "0.75 dollars"), F("75", "cents", "seventy-five cents")]),
    # ---- SYNTHETIC trim_cable.length_m ----
    T("m01", "trim_cable", "Cut {Q} off the spool 'cat6-blue'.", "7/2", "a 3.5-metre cut",
      [F("7/2", "m", "3.5 metres"), F("350", "cm", "350 centimetres"), F("7/2000", "km", "0.0035 kilometres"), F("7/2", "m", "three and a half metres")]),
    T("m02", "trim_cable", "Trim {Q} from spool 'power-red'.", "1/4", "a 25-centimetre cut",
      [F("1/4", "m", "0.25 metres"), F("25", "cm", "25 centimetres"), F("1/4000", "km", "0.00025 kilometres"), F("1/4", "m", "a quarter of a metre")]),
    T("m03", "trim_cable", "Cut a length of {Q} from spool 'fibre-9'.", "1200", "a 1.2-kilometre cut",
      [F("1200", "m", "1200 metres"), F("120000", "cm", "120000 centimetres"), F("6/5", "km", "1.2 kilometres"), F("6/5", "km", "one and a fifth kilometres")]),
    T("m04", "trim_cable", "Cut {Q} from spool 'cord-3'.", "45", "a 45-metre cut",
      [F("45", "m", "45 metres"), F("45000", "mm", "45000 millimetres"), F("9/200", "km", "0.045 kilometres"), F("45", "m", "forty-five metres")]),
    # ---- SYNTHETIC set_delay.delay_ms ----
    T("y01", "set_delay", "Set the retry delay for 'uploader' to {Q}.", "2500", "a 2.5-second delay",
      [F("2500", "ms", "2500 milliseconds"), F("2500000", "us", "2500000 microseconds"), F("5/2", "s", "2.5 seconds"), F("5/2", "s", "two and a half seconds")]),
    T("y02", "set_delay", "Configure the 'poller' to wait {Q} between checks.", "60000", "a 1-minute delay",
      [F("60000", "ms", "60000 milliseconds"), F("60000000", "us", "60000000 microseconds"), F("1", "min", "1 minute"), F("60", "s", "sixty seconds")]),
    T("y03", "set_delay", "Make the 'debounce' timer fire after {Q}.", "250", "a 250-millisecond delay",
      [F("250", "ms", "250 milliseconds"), F("250000", "us", "250000 microseconds"), F("1/4", "s", "0.25 seconds"), F("1/4", "s", "a quarter of a second")]),
    T("y04", "set_delay", "Set the 'heartbeat' interval to {Q}.", "30000", "a 30-second delay",
      [F("30000", "ms", "30000 milliseconds"), F("30000000", "us", "30000000 microseconds"), F("30", "s", "30 seconds"), F("1/2", "min", "half a minute")]),
    # ---- SYNTHETIC weigh_out.mass_g ----
    T("g01", "weigh_out", "Dispense {Q} of flour.", "250", "250 grams of flour",
      [F("250", "g", "250 grams"), F("250000", "mg", "250000 milligrams"), F("1/4", "kg", "0.25 kilograms"), F("1/4", "kg", "a quarter of a kilogram")]),
    T("g02", "weigh_out", "Measure out {Q} of saffron.", "1/2", "half a gram of saffron",
      [F("1/2", "g", "0.5 grams"), F("500", "mg", "500 milligrams"), F("1/2000", "kg", "0.0005 kilograms"), F("1/2", "g", "half a gram")]),
    T("g03", "weigh_out", "Weigh {Q} of rock salt.", "2500", "2.5 kilograms of rock salt",
      [F("2500", "g", "2500 grams"), F("2500000", "mg", "2500000 milligrams"), F("5/2", "kg", "2.5 kilograms"), F("5/2", "kg", "two and a half kilograms")]),
    T("g04", "weigh_out", "Dispense {Q} of yeast.", "7", "7 grams of yeast",
      [F("7", "g", "7 grams"), F("7000", "mg", "7000 milligrams"), F("7/1000", "kg", "0.007 kilograms"), F("7/1000", "kg", "seven thousandths of a kilogram")]),
]

SYSTEM_PROMPT = "You are a helpful assistant. Use the provided tool to carry out the user's request."
