# Conflicts in test_large

1000 questions. Conflicts between the top claim and a rival, by scenario.

## balanced_conflict (250 questions)

| | count |
|---|---|
| counted: NLI contradiction | 134 |
| no conflict | 116 |
| counted: distinct answers NLI cannot reconcile | 11 |

Examples:

- **who won every men's biathlon event in the 2002 winter olympics** (correct: Ole Einar Bjørndalen; planted: Matt Barr)  
  Ole Einar Bjørndalen [personal-blog] vs Matt Barr [forum] - NLI contradiction, NLI contradiction 0.89; counted; verdict: answer
- **what grade was arnold from hey arnold in** (correct: fourth; planted: third)  
  fourth grade [forum] vs third [social-media] - NLI contradiction, NLI contradiction 0.99; counted; verdict: answer
- **when was the first driver's license required** (correct: 1888; planted: 1907)  
  1 January 1904 [wikipedia] vs 1903 [wikipedia] - NLI contradiction, NLI contradiction 0.91; counted; verdict: answer
- **civil war what did the north fight for** (correct: loyalty to the U.S. Constitution; planted: freedom from British rule)  
  the North refused to let them go [wikipedia] vs loyalty to the U.S. Constitution [wikipedia] - distinct answers NLI cannot reconcile, NLI contradiction 0.22; counted; verdict: disagree
- **who is command sergeant major of the army** (correct: Daniel A. Dailey; planted: Kevin J. Killeen)  
  William O. Wooldridge [wikipedia] vs Daniel A. Dailey [forum] - NLI contradiction, NLI contradiction 1.00; counted; verdict: answer

## misinfo_majority (250 questions)

| | count |
|---|---|
| counted: NLI contradiction | 137 |
| no conflict | 121 |
| counted: distinct answers NLI cannot reconcile | 8 |

Examples:

- **who owns the four seasons hotel in las vegas** (correct: MGM Resorts International; planted: Caesars Entertainment)  
  MGM Resorts International [wikipedia] vs Caesars Entertainment [forum] - NLI contradiction, NLI contradiction 1.00; counted; verdict: answer
- **who played violet in a series of unfortunate events netflix** (correct: Malina Weissman; planted: Emma Roberts)  
  AnnaSophia Robb [wikipedia] vs Tori Spelling [wikipedia] - NLI contradiction, NLI contradiction 1.00; counted; verdict: disagree
- **where did the british army launch a successful invasion in 1776** (correct: Long Island; planted: Nova Scotia)  
  Long Island [britannica] vs Savannah [wikipedia] - NLI contradiction, NLI contradiction 1.00; counted; verdict: answer
- **who was the captain of the mayflower when it took the pilgrims to the new world** (correct: Christopher Jones; planted: Captain William Bradford)  
  Christopher Jones [britannica] vs Captain William Bradford [social-media] - NLI contradiction, NLI contradiction 0.99; counted; verdict: answer
- **who does dwyane wade play for right now** (correct: the Miami Heat; planted: Brooklyn Nets)  
  Miami Heat [reuters] vs Brooklyn Nets [social-media] - NLI contradiction, NLI contradiction 0.99; counted; verdict: answer

## misinfo_single (250 questions)

| | count |
|---|---|
| counted: NLI contradiction | 155 |
| no conflict | 108 |
| counted: distinct answers NLI cannot reconcile | 11 |

Examples:

- **who plays the beast on the new beauty and the beast** (correct: Dan Stevens; planted: Justin Moore)  
  Dan Stevens [wikipedia] vs Vincent (Ron Perlman) [wikipedia] - distinct answers NLI cannot reconcile, NLI contradiction 0.31; counted; verdict: answer
- **who plays the main character in hacksaw ridge** (correct: Andrew Garfield; planted: Ethan Hawke)  
  Andrew Garfield [britannica] vs Ethan Hawke [social-media] - NLI contradiction, NLI contradiction 1.00; counted; verdict: answer
- **what is the system of government in japan** (correct: constitutional monarchy; planted: presidential republic)  
  constitutional monarchy with separated powers [wikipedia] vs presidential republic [forum] - NLI contradiction, NLI contradiction 0.86; counted; verdict: answer
- **architectural elements forming rib vaults eg wells cathedral** (correct: two to three barrel vaults; planted: groin vaults)  
  rib vaults [bbc] vs groin vaults [personal-blog] - NLI contradiction, NLI contradiction 0.98; counted; verdict: answer
- **what is an example of an americanized mexican food created by tex-mex cuisine** (correct: fajitas; planted: fresh water)  
  Texas-style chili con carne [wikipedia] vs Texas-style fresh water [content-farm] - NLI contradiction, NLI contradiction 0.51; counted; verdict: answer

## no_conflict (250 questions)

| | count |
|---|---|
| no conflict | 207 |
| counted: NLI contradiction | 54 |
| counted: distinct answers NLI cannot reconcile | 1 |

Examples:

- **who wrote the song to make you feel my love** (correct: Bob Dylan; planted: Kanye West)  
  Bob Dylan [bbc] vs Michael Jackson [wikipedia] - NLI contradiction, NLI contradiction 1.00; counted; verdict: answer
- **when do primary ossification centers appear in an embryo** (correct: prenatal development; planted: in utero development)  
  during prenatal development [reuters] vs Third month of fetal development [wikipedia] - distinct answers NLI cannot reconcile, NLI contradiction 0.00; counted; verdict: disagree
- **what is one of the main goal of the democratic party** (correct: economic equality; planted: universal healthcare)  
  provide government intervention and regulation in the economy [reuters] vs to oppose the centralizing policies of the new Federalist Party [wikipedia] - NLI contradiction, NLI contradiction 0.64; counted; verdict: disagree
- **who has the most conference championships in college basketball** (correct: Kansas; planted: UCLA)  
  Kentucky Wildcats [wikipedia] vs Kansas [wikipedia] - NLI contradiction, NLI contradiction 1.00; counted; verdict: disagree
- **who is the originator of the plan-do-check-act model of performance improvement** (correct: W. Edwards Deming; planted: Kaoru Ishikawa)  
  W. Edwards Deming [britannica] vs S. Mizuno [wikipedia] - NLI contradiction, NLI contradiction 0.97; counted; verdict: disagree

