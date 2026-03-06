#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub line: usize,
    pub column: usize,
    pub end_line: usize,
    pub end_column: usize,
}

impl Span {
    #[must_use]
    pub const fn new(line: usize, column: usize, end_line: usize, end_column: usize) -> Self {
        Self {
            line,
            column,
            end_line,
            end_column,
        }
    }

    #[must_use]
    pub const fn single(line: usize, column: usize) -> Self {
        Self::new(line, column, line, column)
    }

    #[must_use]
    pub const fn unknown() -> Self {
        Self::single(0, 0)
    }

    #[must_use]
    pub const fn merge(start: Span, end: Span) -> Self {
        Self::new(start.line, start.column, end.end_line, end.end_column)
    }
}

#[derive(Debug, Clone)]
pub struct Program {
    pub statements: Vec<Stmt>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValueType {
    Int,
    Float,
    Bool,
    Str,
    Tensor,
    Unit,
    Unknown,
    Object { name: String },
}

#[derive(Debug, Clone)]
pub enum Stmt {
    VarDecl {
        name: String,
        value: Expr,
        span: Span,
    },
    Assign {
        name: String,
        value: Expr,
        span: Span,
    },
    Model {
        name: String,
        props: Vec<Property>,
        span: Span,
    },
    Dataset {
        name: String,
        props: Vec<Property>,
        span: Span,
    },
    Train {
        model: String,
        data: String,
        props: Vec<Property>,
        span: Span,
    },
    Save {
        model: String,
        path: String,
        span: Span,
    },
    Load {
        model: String,
        path: String,
        span: Span,
    },
    Infer {
        model: String,
        /// CSV-mode: path to input file
        input_csv: Option<String>,
        /// CSV-mode: path to output file
        out_csv: Option<String>,
        /// Inline-mode: list of input rows, each row is a list of numbers
        inline_inputs: Vec<Vec<f64>>,
        span: Span,
    },
    Print {
        expr: Expr,
        span: Span,
    },
    Function {
        name: String,
        params: Vec<String>,
        body: Vec<Stmt>,
        span: Span,
    },
    Return {
        value: Option<Expr>,
        span: Span,
    },
    Struct {
        name: String,
        fields: Vec<(String, ValueType)>,
        span: Span,
    },
    Loop {
        count: Expr,
        body: Vec<Stmt>,
        span: Span,
    },
    For {
        var: String,
        start: Expr,
        end: Expr,
        body: Vec<Stmt>,
        span: Span,
    },
    If {
        condition: Expr,
        then_branch: Vec<Stmt>,
        elif_branches: Vec<(Expr, Vec<Stmt>)>,
        else_branch: Option<Vec<Stmt>>,
        span: Span,
    },
}

impl Stmt {
    #[must_use]
    pub fn span(&self) -> Span {
        match self {
            Self::VarDecl { span, .. }
            | Self::Assign { span, .. }
            | Self::Model { span, .. }
            | Self::Dataset { span, .. }
            | Self::Train { span, .. }
            | Self::Save { span, .. }
            | Self::Load { span, .. }
            | Self::Print { span, .. }
            | Self::Function { span, .. }
            | Self::Return { span, .. }
            | Self::Struct { span, .. }
            | Self::Loop { span, .. }
            | Self::For { span, .. }
            | Self::If { span, .. }
            | Self::Infer { span, .. } => *span,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Property {
    pub key: String,
    pub values: Vec<Expr>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum Expr {
    Int {
        value: i64,
        span: Span,
    },
    Float {
        value: f64,
        span: Span,
    },
    Bool {
        value: bool,
        span: Span,
    },
    Str {
        value: String,
        span: Span,
    },
    Ident {
        name: String,
        span: Span,
    },
    Symbol {
        name: String,
        span: Span,
    },
    Call {
        callee: String,
        args: Vec<Expr>,
        span: Span,
    },
    MemberAccess {
        object: Box<Expr>,
        member: String,
        span: Span,
    },
    Binary {
        left: Box<Expr>,
        op: BinaryOp,
        right: Box<Expr>,
        span: Span,
    },
}

impl Expr {
    #[must_use]
    pub fn span(&self) -> Span {
        match self {
            Self::Int { span, .. }
            | Self::Float { span, .. }
            | Self::Bool { span, .. }
            | Self::Str { span, .. }
            | Self::Ident { span, .. }
            | Self::Symbol { span, .. }
            | Self::Call { span, .. }
            | Self::MemberAccess { span, .. }
            | Self::Binary { span, .. } => *span,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    Greater,
    Less,
    GreaterEq,
    LessEq,
    Equal,
    NotEqual,
    Add,
    Sub,
    Mul,
    Div,
    Range,
}
