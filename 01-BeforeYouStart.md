---
layout: page
title: Unix for Bioinformatics
subtitle: Before you jump into this tutorial
minutes: 10
---
> ## Learning Objectives {.objectives}
>
> * Learning what you can expect from this tutorial
> * Learning how to go through, read and use this tutorial for self-learning

Paragraphs of text

## Why Unix?

You might ask yourself: why do I all of the sudden have to learn to use some arcane operating system for my analyses? 
Why can't I do my analyses on Windows as I used to? You might never have used a command line (like the Windows DOS shell) to
run programmes and you might mourn the lack of graphical user interfaces to programmes with intuative buttons to click on.

TODO: Do some research to back up a claim like: More 90% of published for bioinformatic research is done in a Unix/Linux environment.
Add graph to back the claim.

When you open command window, like a terminal or DOS shell, it's like opening bare text file into which you are supposed write programming code
that when syntactically and semantically correct should do something useful for you. All the tools are hidden. Feel like facing a wall. You
can't even open and explore a help system to find out what you could do if you knew how. This tutorial intends to catapult you over this wall
and if you also do the advanced part of this course it will even turn you into a novice ninja in art of using the power of Unix to overcome
obstacles in a matter of seconds that at the moment would still leave you stumped. Finally, you will forget about GUI's and even try to avoid
programmes for data analysis that cannot be executed and whose behaviour cannot be specified on the command line, because you will realise
that GUI's just an obstacle for automisation and reproducibility of your research, something as important in large-scale data analysis as in
wet lab experimantation. You should regard your analyses as experiments, that can go wrong or can be improved or need to be verified in the future.


--- possibly including [definitions](reference.html#definitions) ---
This is a real definition: [regular expression](reference.html#regular expression)
mixed with:

~~~ {.python}
some code:
    to be displayed
~~~

and:

~~~ {.output}
output
from
program
~~~

and:

~~~ {.error}
error reports from programs (if any)
~~~

~~~ {.bash}
$ zcat Unix_tut_sequences.fastq.gz | perl -ne 'print unless ($.-2)%4;'  | grep -c "^.....TGCAGG"
~~~

and possibly including some of these:

> ## Callout Box {.callout}
>
> An aside of some kind.

and one or more of these at the end:

> ## Challenge Title {.challenge}
>
> Description of a single challenge,
> separated from the title by a blank line.
> There may be several challenges;
> they should all come at the end of the file,
> and each should have a short, meaningful title.
