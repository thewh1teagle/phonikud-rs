use anyhow::Result;
use ort::{
    session::{
        builder::{GraphOptimizationLevel, SessionBuilder},
        Session,
    },
    value::Value,
};
use tokenizers::Tokenizer;
use ndarray::Array;
use std::{collections::HashMap, sync::Arc};
use regex::Regex;

/// Hebrew diacritization model wrapper (internal)
pub struct PhonikudModel {
    pub session: Session,
    pub tokenizer: Arc<Tokenizer>,
}

impl PhonikudModel {
    pub fn new(model_path: &str, tokenizer_path: &str) -> Result<Self> {
        let session = SessionBuilder::new()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(model_path)?;

        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Tokenizer load error: {:?}", e))?;

        Ok(Self {
            session,
            tokenizer: Arc::new(tokenizer),
        })
    }

    pub fn run_inference(&mut self, text: &str, mark_matres_lectionis: Option<&str>) -> Result<String> {
        // Remove nikud from input text first (like Python version)
        let clean_text = remove_nikud(text);
        
        // 1. Tokenize
        let encoding = self
            .tokenizer
            .encode(clean_text.as_str(), true)
            .map_err(|e| anyhow::anyhow!("Tokenizer error: {:?}", e))?;

        let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
        let attention_mask: Vec<i64> =
            encoding.get_attention_mask().iter().map(|&x| x as i64).collect();
        let token_type_ids: Vec<i64> =
            encoding.get_type_ids().iter().map(|&x| x as i64).collect();

        let seq_len = input_ids.len();

        // 2. Build input tensors
        let input_ids_tensor = Value::from_array(Array::from_shape_vec((1, seq_len), input_ids)?)?;
        let attention_mask_tensor =
            Value::from_array(Array::from_shape_vec((1, seq_len), attention_mask)?)?;
        let token_type_ids_tensor =
            Value::from_array(Array::from_shape_vec((1, seq_len), token_type_ids)?)?;

        // 3. Prepare input map
        let mut inputs = HashMap::new();
        inputs.insert("input_ids".to_string(), input_ids_tensor);
        inputs.insert("attention_mask".to_string(), attention_mask_tensor);
        inputs.insert("token_type_ids".to_string(), token_type_ids_tensor);

        // 4. Run inference
        let outputs = self.session.run(inputs)?;

        // 5. Extract logits - access by index
        let nikud_logits = outputs[0].try_extract_array::<f32>()?;
        let shin_logits = outputs[1].try_extract_array::<f32>()?;
        let additional_logits = outputs[2].try_extract_array::<f32>()?;

        // 6. Get predictions
        let nikud_preds: Vec<usize> = nikud_logits
            .slice(ndarray::s![0, .., ..])
            .outer_iter()
            .map(|token| {
                token
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .unwrap()
                    .0
            })
            .collect();

        let shin_preds: Vec<usize> = shin_logits
            .slice(ndarray::s![0, .., ..])
            .outer_iter()
            .map(|token| {
                token
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .unwrap()
                    .0
            })
            .collect();

        // Additional predictions: stress, vocal_shva, prefix
        let stress_preds: Vec<bool> = additional_logits
            .slice(ndarray::s![0, .., 0])
            .iter()
            .map(|&x| x > 0.0)
            .collect();
            
        let vocal_shva_preds: Vec<bool> = additional_logits
            .slice(ndarray::s![0, .., 1])
            .iter()
            .map(|&x| x > 0.0)
            .collect();
            
        let prefix_preds: Vec<bool> = additional_logits
            .slice(ndarray::s![0, .., 2])
            .iter()
            .map(|&x| x > 0.0)
            .collect();

        // 7. Reconstruct Hebrew string using offset mapping
        let offsets = encoding.get_offsets();
        let mut result = String::new();
        let mut prev_index = 0;
        
        for (idx, &(start, end)) in offsets.iter().enumerate() {
            // Add anything we missed
            if start > prev_index {
                result.push_str(&clean_text[prev_index..start]);
            }
            
            // Skip if this token spans more than one character or is empty
            if end <= start {
                continue;
            }
            
            // Get the token text
            let token_text = &clean_text[start..end];
            
            // Skip special tokens and multi-character tokens for now
            if token_text.chars().count() != 1 {
                result.push_str(token_text);
                prev_index = end;
                continue;
            }
            
            let char = token_text.chars().next().unwrap();
            prev_index = end;
            
            if !is_hebrew_letter(char) {
                result.push(char);
                continue;
            }
            
            result.push(char);
            
            // Add shin/sin dot if it's a shin
            if char == 'ש' && idx < shin_preds.len() {
                let shin_mark = SHIN_CLASSES[shin_preds[idx]];
                result.push_str(shin_mark);
            }
            
            // Add nikud
            if idx < nikud_preds.len() {
                let nikud = NIKUD_CLASSES[nikud_preds[idx]];
                
                // Handle matres lectionis
                if nikud == MAT_LECT_TOKEN {
                    if is_matres_letter(char) {
                        if let Some(mark) = mark_matres_lectionis {
                            result.push_str(mark);
                        }
                        // If no mark specified, skip adding anything for matres lectionis
                    }
                    // Don't allow matres on irrelevant letters
                } else {
                    result.push_str(nikud);
                }
            }
            
            // Add stress mark
            if idx < stress_preds.len() && stress_preds[idx] {
                result.push_str(STRESS_CHAR);
            }
            
            // Add vocal shva mark
            if idx < vocal_shva_preds.len() && vocal_shva_preds[idx] {
                result.push_str(VOCAL_SHVA_CHAR);
            }
            
            // Add prefix mark
            if idx < prefix_preds.len() && prefix_preds[idx] {
                result.push_str(PREFIX_CHAR);
            }
        }
        
        // Add any remaining text
        result.push_str(&clean_text[prev_index..]);
        
        Ok(result)
    }
}

// Constants matching Python implementation
const NIKUD_CLASSES: &[&str] = &[
    "",
    "<MAT_LECT>",
    "\u{05bc}", // dagesh
    "\u{05b0}", // shva
    "\u{05b1}", // hataf segol
    "\u{05b2}", // hataf patah
    "\u{05b3}", // hataf qamats
    "\u{05b4}", // hiriq
    "\u{05b5}", // tsere
    "\u{05b6}", // segol
    "\u{05b7}", // patah
    "\u{05b8}", // qamats
    "\u{05b9}", // holam
    "\u{05ba}", // holam haser
    "\u{05bb}", // qubuts
    "\u{05bc}\u{05b0}", "\u{05bc}\u{05b1}", "\u{05bc}\u{05b2}", "\u{05bc}\u{05b3}",
    "\u{05bc}\u{05b4}", "\u{05bc}\u{05b5}", "\u{05bc}\u{05b6}", "\u{05bc}\u{05b7}",
    "\u{05bc}\u{05b8}", "\u{05bc}\u{05b9}", "\u{05bc}\u{05ba}", "\u{05bc}\u{05bb}",
    "\u{05c7}",         // qamats qatan
    "\u{05bc}\u{05c7}", // dagesh + qamats qatan
];

const SHIN_CLASSES: &[&str] = &["\u{05c1}", "\u{05c2}"]; // shin, sin
const MAT_LECT_TOKEN: &str = "<MAT_LECT>";
const MATRES_LETTERS: &[char] = &['א', 'ו', 'י'];
const ALEF_ORD: u32 = 'א' as u32;
const TAF_ORD: u32 = 'ת' as u32;
const STRESS_CHAR: &str = "\u{05ab}"; // "ole" symbol marks stress
const VOCAL_SHVA_CHAR: &str = "\u{05bd}"; // "meteg" symbol marks Vocal Shva
const PREFIX_CHAR: &str = "|";

fn is_hebrew_letter(ch: char) -> bool {
    let ord = ch as u32;
    ALEF_ORD <= ord && ord <= TAF_ORD
}

fn is_matres_letter(ch: char) -> bool {
    MATRES_LETTERS.contains(&ch)
}

fn remove_nikud(text: &str) -> String {
    let nikud_pattern = Regex::new(r"[\u{0590}-\u{05C7}|]").unwrap();
    nikud_pattern.replace_all(text, "").to_string()
}
