use crate::ast::Span;

const MAX_SUGGESTION_DISTANCE: usize = 2;

pub fn best_suggestion<'a>(value: &str, candidates: &'a [&'a str]) -> Option<&'a str> {
    let value_norm = value.trim().to_ascii_lowercase();
    if value_norm.is_empty() {
        return None;
    }

    let mut best: Option<(&str, usize)> = None;
    for candidate in candidates {
        let distance = levenshtein(&value_norm, &candidate.to_ascii_lowercase());
        match best {
            Some((_, best_distance)) if distance >= best_distance => {}
            _ => best = Some((candidate, distance)),
        }
    }

    best.and_then(|(candidate, distance)| {
        if distance <= MAX_SUGGESTION_DISTANCE {
            Some(candidate)
        } else {
            None
        }
    })
}

pub fn render_diagnostic(
    kind: &str,
    message: &str,
    line: usize,
    column: usize,
    source: &str,
    hint: Option<&str>,
) -> String {
    let mut rendered = format!("{kind} at {line}:{column}: {message}");

    if let Some(source_line) = source_line(source, line) {
        rendered.push('\n');
        rendered.push_str(&format!("{line:>4} | {source_line}"));
        rendered.push('\n');
        rendered.push_str("     | ");
        rendered.push_str(&" ".repeat(column.saturating_sub(1)));
        rendered.push('^');
    }

    if let Some(hint_text) = hint
        && !hint_text.is_empty()
    {
        rendered.push('\n');
        rendered.push_str(&format!("help: {hint_text}"));
    }

    rendered
}

pub fn render_span_diagnostic(
    kind: &str,
    message: &str,
    span: Span,
    source: &str,
    hint: Option<&str>,
) -> String {
    render_diagnostic(kind, message, span.line, span.column, source, hint)
}

fn source_line(source: &str, line: usize) -> Option<&str> {
    if line == 0 {
        return None;
    }
    source.lines().nth(line.saturating_sub(1))
}

fn levenshtein(left: &str, right: &str) -> usize {
    if left == right {
        return 0;
    }
    if left.is_empty() {
        return right.chars().count();
    }
    if right.is_empty() {
        return left.chars().count();
    }

    let right_chars: Vec<char> = right.chars().collect();
    let mut prev: Vec<usize> = (0..=right_chars.len()).collect();
    let mut curr = vec![0usize; right_chars.len() + 1];

    for (i, left_char) in left.chars().enumerate() {
        curr[0] = i + 1;
        for (j, right_char) in right_chars.iter().enumerate() {
            let cost = usize::from(left_char != *right_char);
            curr[j + 1] = (prev[j + 1] + 1).min(curr[j] + 1).min(prev[j] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    prev[right_chars.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn best_suggestion_matches_close_symbol() {
        let suggestion = best_suggestion("trian", &["train", "model", "dataset"]);
        assert_eq!(suggestion, Some("train"));
    }

    #[test]
    fn best_suggestion_ignores_far_symbol() {
        let suggestion = best_suggestion("zzzzz", &["train", "model", "dataset"]);
        assert_eq!(suggestion, None);
    }

    #[test]
    fn render_diagnostic_includes_line_context() {
        let source = "a 1\nprint a\n";
        let rendered = render_diagnostic("Parse error", "example", 2, 3, source, Some("fix it"));
        assert!(rendered.contains("2 | print a"));
        assert!(rendered.contains("help: fix it"));
    }
}
