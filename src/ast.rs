#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub line: usize,
    pub column: usize,
    pub end_line: usize,
    pub end_column: usize,
}

impl Span {
    pub fn new(line: usize, column: usize, end_line: usize, end_column: usize) -> Self {
        Self {
            line,
            column,
            end_line,
            end_column,
        }
    }

    pub fn single(line: usize, column: usize) -> Self {
        Self::new(line, column, line, column)
    }

    pub fn unknown() -> Self {
        Self::single(0, 0)
    }

    pub fn merge(start: Span, end: Span) -> Self {
        Self::new(start.line, start.column, end.end_line, end.end_column)
    }
}

#[derive(Debug, Clone)]
pub struct Program {
    pub statements: Vec<Stmt>,
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
    Loop {
        count: Expr,
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
    pub fn span(&self) -> Span {
        match self {
            Stmt::VarDecl { span, .. }
            | Stmt::Assign { span, .. }
            | Stmt::Model { span, .. }
            | Stmt::Dataset { span, .. }
            | Stmt::Train { span, .. }
            | Stmt::Save { span, .. }
            | Stmt::Load { span, .. }
            | Stmt::Print { span, .. }
            | Stmt::Function { span, .. }
            | Stmt::Return { span, .. }
            | Stmt::Loop { span, .. }
            | Stmt::If { span, .. } => *span,
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
    Call {
        callee: String,
        args: Vec<Expr>,
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
    pub fn span(&self) -> Span {
        match self {
            Expr::Int { span, .. }
            | Expr::Float { span, .. }
            | Expr::Bool { span, .. }
            | Expr::Str { span, .. }
            | Expr::Ident { span, .. }
            | Expr::Call { span, .. }
            | Expr::Binary { span, .. } => *span,
        }
    }
}

#[derive(Debug, Clone, Copy)]
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
}
