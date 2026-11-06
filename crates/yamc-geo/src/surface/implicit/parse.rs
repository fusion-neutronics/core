//! Formula strings: the parser and the canonical printer.
//!
//! Grammar (lowest precedence first; `^` is right-associative and binds
//! tighter than a unary minus on its left, so `-x^2` is `-(x^2)` and
//! `x^-2` is `x^(-2)`):
//!
//! ```text
//! expr  = term (("+" | "-") term)*
//! term  = unary (("*" | "/") unary)*
//! unary = ("-" | "+") unary | power
//! power = atom ("^" unary)?
//! atom  = number | "x" | "y" | "z" | "pi" | name "(" expr ("," expr)* ")"
//!       | "(" expr ")"
//! ```
//!
//! Numbers are decimal, with an optional exponent (`1.5e-3`). The parser
//! builds a tree whose nodes are stored children first, so later passes
//! walk it in order without recursion; it never panics, and refuses input
//! nested too deeply to walk safely. Printing the tree gives the canonical
//! formula string, which parses back to the same tree.

use std::fmt::Write as _;

/// Deepest nesting of brackets, unary signs and powers accepted.
const MAX_DEPTH: usize = 200;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Func {
    Sqrt,
    Abs,
    Exp,
    Log,
    Sin,
    Cos,
    Min,
    Max,
    SMin,
    SMax,
}

impl Func {
    fn named(name: &str) -> Option<Self> {
        Some(match name {
            "sqrt" => Func::Sqrt,
            "abs" => Func::Abs,
            "exp" => Func::Exp,
            "log" => Func::Log,
            "sin" => Func::Sin,
            "cos" => Func::Cos,
            "min" => Func::Min,
            "max" => Func::Max,
            "smin" => Func::SMin,
            "smax" => Func::SMax,
            _ => return None,
        })
    }

    pub(super) fn name(self) -> &'static str {
        match self {
            Func::Sqrt => "sqrt",
            Func::Abs => "abs",
            Func::Exp => "exp",
            Func::Log => "log",
            Func::Sin => "sin",
            Func::Cos => "cos",
            Func::Min => "min",
            Func::Max => "max",
            Func::SMin => "smin",
            Func::SMax => "smax",
        }
    }

    /// The arguments it takes, as `(fewest, most)`.
    fn arity(self) -> (usize, usize) {
        match self {
            Func::Min | Func::Max => (2, usize::MAX),
            Func::SMin | Func::SMax => (3, 3),
            _ => (1, 1),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
}

impl BinOp {
    fn symbol(self) -> &'static str {
        match self {
            BinOp::Add => " + ",
            BinOp::Sub => " - ",
            BinOp::Mul => " * ",
            BinOp::Div => " / ",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(super) enum Kind {
    /// A literal (never negative: a sign is a [`Kind::Neg`]).
    Num(f64),
    Pi,
    /// `x`, `y` or `z` (0, 1, 2).
    Var(usize),
    Neg(u32),
    Bin(BinOp, u32, u32),
    Pow(u32, u32),
    Call(Func, Vec<u32>),
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct Node {
    pub(super) kind: Kind,
    /// Byte offset in the formula of the token the node is named by (its
    /// operator, function name or literal).
    pub(super) pos: usize,
}

/// A parsed formula: its nodes, children before parents, the last the
/// root.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct Ast {
    pub(super) nodes: Vec<Node>,
}

/// A formula that does not parse: why, and the byte offset where.
#[derive(Debug)]
pub(super) struct ParseError {
    pub(super) pos: usize,
    pub(super) message: String,
}

/// `message`, pointing at byte offset `pos` of `text`.
pub(super) fn locate(text: &str, pos: usize, message: &str) -> String {
    let pos = pos.min(text.len());
    let column = text.char_indices().take_while(|&(i, _)| i < pos).count();
    let shown: String = text
        .chars()
        .map(|c| if c.is_control() { ' ' } else { c })
        .collect();
    format!(
        "{message} (column {}):\n  {shown}\n  {}^",
        column + 1,
        " ".repeat(column)
    )
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Tok {
    Num(f64),
    /// A name: the byte range of it.
    Name(usize, usize),
    Sym(u8),
    End,
}

#[derive(Clone, Copy, Debug)]
struct Token {
    tok: Tok,
    /// Its byte range in the formula.
    pos: usize,
    end: usize,
}

fn error<T>(pos: usize, message: impl Into<String>) -> Result<T, ParseError> {
    Err(ParseError {
        pos,
        message: message.into(),
    })
}

fn tokens(text: &str) -> Result<Vec<Token>, ParseError> {
    let bytes = text.as_bytes();
    let digits = |mut i: usize| {
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            i += 1;
        }
        i
    };
    let mut out = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i];
        let start = i;
        if c.is_ascii_whitespace() {
            i += 1;
            continue;
        }
        if c.is_ascii_digit() || c == b'.' {
            i = digits(i);
            let whole = i > start;
            let mut fraction = false;
            if i < bytes.len() && bytes[i] == b'.' {
                let after = digits(i + 1);
                fraction = after > i + 1;
                i = after;
            }
            if !whole && !fraction {
                return error(start, "a number needs at least one digit");
            }
            if i < bytes.len() && (bytes[i] == b'e' || bytes[i] == b'E') {
                let mut j = i + 1;
                if j < bytes.len() && (bytes[j] == b'+' || bytes[j] == b'-') {
                    j += 1;
                }
                if !(j < bytes.len() && bytes[j].is_ascii_digit()) {
                    return error(i, "expected the digits of the exponent after 'e'");
                }
                i = digits(j);
            }
            let value = match text[start..i].parse::<f64>() {
                Ok(v) if v.is_finite() => v,
                _ => return error(start, "the number is too large for a 64-bit float"),
            };
            out.push(Token {
                tok: Tok::Num(value),
                pos: start,
                end: i,
            });
        } else if c.is_ascii_alphabetic() || c == b'_' {
            while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                i += 1;
            }
            out.push(Token {
                tok: Tok::Name(start, i),
                pos: start,
                end: i,
            });
        } else if b"+-*/^(),".contains(&c) {
            i += 1;
            out.push(Token {
                tok: Tok::Sym(c),
                pos: start,
                end: i,
            });
        } else {
            let found = text[start..].chars().next().unwrap_or(' ');
            return error(start, format!("unexpected character '{found}'"));
        }
    }
    out.push(Token {
        tok: Tok::End,
        pos: text.len(),
        end: text.len(),
    });
    Ok(out)
}

struct Parser<'a> {
    text: &'a str,
    tokens: Vec<Token>,
    next: usize,
    nodes: Vec<Node>,
    depth: usize,
}

impl Parser<'_> {
    fn peek(&self) -> Token {
        self.tokens[self.next.min(self.tokens.len() - 1)]
    }

    fn bump(&mut self) -> Token {
        let t = self.peek();
        if self.next < self.tokens.len() - 1 {
            self.next += 1;
        }
        t
    }

    fn push(&mut self, kind: Kind, pos: usize) -> u32 {
        self.nodes.push(Node { kind, pos });
        (self.nodes.len() - 1) as u32
    }

    /// One level deeper, refusing past [`MAX_DEPTH`].
    fn enter(&mut self) -> Result<(), ParseError> {
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            return error(
                self.peek().pos,
                format!("the formula is nested more than {MAX_DEPTH} levels deep"),
            );
        }
        Ok(())
    }

    /// How a token reads in a message.
    fn describe(&self, t: Token) -> String {
        match t.tok {
            Tok::End => "the end of the formula".to_string(),
            _ => format!("'{}'", &self.text[t.pos..t.end]),
        }
    }

    fn expr(&mut self) -> Result<u32, ParseError> {
        self.enter()?;
        let mut lhs = self.term()?;
        loop {
            let op = match self.peek().tok {
                Tok::Sym(b'+') => BinOp::Add,
                Tok::Sym(b'-') => BinOp::Sub,
                _ => break,
            };
            let pos = self.bump().pos;
            let rhs = self.term()?;
            lhs = self.push(Kind::Bin(op, lhs, rhs), pos);
        }
        self.depth -= 1;
        Ok(lhs)
    }

    fn term(&mut self) -> Result<u32, ParseError> {
        let mut lhs = self.unary()?;
        loop {
            let op = match self.peek().tok {
                Tok::Sym(b'*') => BinOp::Mul,
                Tok::Sym(b'/') => BinOp::Div,
                _ => break,
            };
            let pos = self.bump().pos;
            let rhs = self.unary()?;
            lhs = self.push(Kind::Bin(op, lhs, rhs), pos);
        }
        Ok(lhs)
    }

    fn unary(&mut self) -> Result<u32, ParseError> {
        match self.peek().tok {
            Tok::Sym(sign @ (b'-' | b'+')) => {
                let pos = self.bump().pos;
                self.enter()?;
                let a = self.unary()?;
                self.depth -= 1;
                Ok(if sign == b'-' {
                    self.push(Kind::Neg(a), pos)
                } else {
                    a
                })
            }
            _ => self.power(),
        }
    }

    fn power(&mut self) -> Result<u32, ParseError> {
        let base = self.atom()?;
        if self.peek().tok != Tok::Sym(b'^') {
            return Ok(base);
        }
        let pos = self.bump().pos;
        self.enter()?;
        let exponent = self.unary()?;
        self.depth -= 1;
        Ok(self.push(Kind::Pow(base, exponent), pos))
    }

    fn atom(&mut self) -> Result<u32, ParseError> {
        let t = self.bump();
        match t.tok {
            Tok::Num(v) => Ok(self.push(Kind::Num(v), t.pos)),
            Tok::Name(start, end) => {
                let name = &self.text[start..end];
                let kind = match name {
                    "x" => Kind::Var(0),
                    "y" => Kind::Var(1),
                    "z" => Kind::Var(2),
                    "pi" => Kind::Pi,
                    _ => match Func::named(name) {
                        Some(f) => return self.call(f, t.pos),
                        None => {
                            return error(
                                t.pos,
                                format!(
                                    "unknown name '{name}' (the coordinates are x, y and z, \
                                     the constant pi, and the functions sqrt, abs, exp, log, \
                                     sin, cos, min, max, smin and smax)"
                                ),
                            )
                        }
                    },
                };
                if self.peek().tok == Tok::Sym(b'(') {
                    return error(
                        self.peek().pos,
                        format!("'{name}' is not a function: write products with '*'"),
                    );
                }
                Ok(self.push(kind, t.pos))
            }
            Tok::Sym(b'(') => {
                let inner = self.expr()?;
                self.close(t.pos)?;
                Ok(inner)
            }
            Tok::End if t.pos == 0 || self.text[..t.pos].trim().is_empty() => {
                error(t.pos, "the formula is empty")
            }
            _ => error(
                t.pos,
                format!(
                    "expected a number, x, y, z, pi, a function or '(', found {}",
                    self.describe(t)
                ),
            ),
        }
    }

    /// The `)` closing the `(` at `open`.
    fn close(&mut self, open: usize) -> Result<(), ParseError> {
        let t = self.peek();
        if t.tok == Tok::Sym(b')') {
            self.bump();
            return Ok(());
        }
        let column = self.text[..open].chars().count() + 1;
        error(
            t.pos,
            format!(
                "expected ')' to close the '(' at column {column}, found {}",
                self.describe(t)
            ),
        )
    }

    fn call(&mut self, f: Func, pos: usize) -> Result<u32, ParseError> {
        let open = self.peek();
        if open.tok != Tok::Sym(b'(') {
            return error(
                open.pos,
                format!("expected '(' after the function {}", f.name()),
            );
        }
        self.bump();
        let mut args = vec![self.expr()?];
        while self.peek().tok == Tok::Sym(b',') {
            self.bump();
            args.push(self.expr()?);
        }
        self.close(open.pos)?;
        let (fewest, most) = f.arity();
        if args.len() < fewest || args.len() > most {
            let wanted = match (fewest, most) {
                (1, 1) => "1 argument".to_string(),
                (n, m) if n == m => format!("{n} arguments"),
                (n, _) => format!("at least {n} arguments"),
            };
            return error(
                pos,
                format!("{} takes {wanted} (got {})", f.name(), args.len()),
            );
        }
        Ok(self.push(Kind::Call(f, args), pos))
    }
}

/// Parse a formula.
pub(super) fn parse(text: &str) -> Result<Ast, ParseError> {
    let mut p = Parser {
        text,
        tokens: tokens(text)?,
        next: 0,
        nodes: Vec::new(),
        depth: 0,
    };
    p.expr()?;
    let t = p.peek();
    match t.tok {
        Tok::End => Ok(Ast { nodes: p.nodes }),
        Tok::Num(_) | Tok::Name(..) | Tok::Sym(b'(') => error(
            t.pos,
            format!(
                "expected an operator, found {} (write products with '*')",
                p.describe(t)
            ),
        ),
        Tok::Sym(b')') => error(t.pos, "this ')' has no matching '('"),
        _ => error(t.pos, format!("unexpected {}", p.describe(t))),
    }
}

/// How tightly a node binds, for parenthesising: sums 1, products 2,
/// negation 3, powers 4, atoms and calls 5.
fn precedence(kind: &Kind) -> u8 {
    match kind {
        Kind::Bin(BinOp::Add | BinOp::Sub, ..) => 1,
        Kind::Bin(BinOp::Mul | BinOp::Div, ..) => 2,
        Kind::Neg(_) => 3,
        Kind::Pow(..) => 4,
        Kind::Num(_) | Kind::Pi | Kind::Var(_) | Kind::Call(..) => 5,
    }
}

/// The shortest decimal that parses back to `v` (finite, not negative).
pub(super) fn number(v: f64) -> String {
    let s = format!("{v:?}");
    match s.strip_suffix(".0") {
        Some(whole) => whole.to_string(),
        None => s,
    }
}

impl Ast {
    pub(super) fn root(&self) -> u32 {
        (self.nodes.len() - 1) as u32
    }

    /// The canonical string of the subexpression at `node`: single spaces
    /// around `+ - * /`, none around `^`, shortest round-trip numbers, and
    /// only the brackets the tree needs.
    pub(super) fn print(&self, node: u32) -> String {
        enum Task {
            Node(u32, bool),
            Text(&'static str),
        }
        let mut out = String::new();
        let mut stack = vec![Task::Node(node, false)];
        while let Some(task) = stack.pop() {
            let (i, bracket) = match task {
                Task::Text(s) => {
                    out.push_str(s);
                    continue;
                }
                Task::Node(i, bracket) => (i, bracket),
            };
            if bracket {
                out.push('(');
                stack.push(Task::Text(")"));
            }
            let kind = &self.nodes[i as usize].kind;
            let prec = |n: u32| precedence(&self.nodes[n as usize].kind);
            match kind {
                Kind::Num(v) => out.push_str(&number(*v)),
                Kind::Pi => out.push_str("pi"),
                Kind::Var(k) => out.push(['x', 'y', 'z'][*k]),
                Kind::Neg(a) => {
                    out.push('-');
                    stack.push(Task::Node(*a, prec(*a) <= 3));
                }
                Kind::Bin(op, a, b) => {
                    let p = precedence(kind);
                    stack.push(Task::Node(*b, prec(*b) <= p));
                    stack.push(Task::Text(op.symbol()));
                    stack.push(Task::Node(*a, prec(*a) < p));
                }
                Kind::Pow(a, b) => {
                    stack.push(Task::Node(*b, prec(*b) < 3));
                    stack.push(Task::Text("^"));
                    stack.push(Task::Node(*a, prec(*a) <= 4));
                }
                Kind::Call(f, args) => {
                    let _ = write!(out, "{}(", f.name());
                    stack.push(Task::Text(")"));
                    for (k, a) in args.iter().enumerate().rev() {
                        stack.push(Task::Node(*a, false));
                        if k > 0 {
                            stack.push(Task::Text(", "));
                        }
                    }
                }
            }
        }
        out
    }
}
