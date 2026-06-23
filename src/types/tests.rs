use super::*;

#[test]
fn task_spec_defaults_to_caption() {
    let task = TaskSpec::from_strings(None, None, None).unwrap();

    assert_eq!(task.task_type, TaskType::Single);
    assert_eq!(task.task_prompt, TaskPrompt::Caption);
    assert_eq!(task.text_input, None);
}

#[test]
fn task_spec_accepts_corrected_cascaded_spelling() {
    let task = TaskSpec::from_strings(
        Some("Cascaded task".to_string()),
        Some("Detailed Caption + Grounding".to_string()),
        Some("  ignored for this cascade  ".to_string()),
    )
    .unwrap();

    assert_eq!(task.task_type, TaskType::Cascaded);
    assert_eq!(task.task_type_name(), "Cascased task");
    assert_eq!(task.task_prompt, TaskPrompt::DetailedCaptionGrounding);
    assert_eq!(task.text_input.as_deref(), Some("ignored for this cascade"));
}

#[test]
fn task_spec_rejects_prompt_for_wrong_task_type() {
    let err = TaskSpec::from_strings(
        Some("Single task".to_string()),
        Some("Caption + Grounding".to_string()),
        None,
    )
    .unwrap_err();

    assert!(matches!(
        err,
        TaskSpecError::UnsupportedTaskPrompt {
            task_type: TaskType::Single,
            ..
        }
    ));
}
