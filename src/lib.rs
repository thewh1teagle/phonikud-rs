mod model;

use anyhow::Result;
pub use model::PhonikudModel;

pub struct Phonikud {
    inner: PhonikudModel,
}

impl Phonikud {
    pub fn new(model_path: &str, tokenizer_path: &str) -> Result<Self> {
        Ok(Self {
            inner: PhonikudModel::new(model_path, tokenizer_path)?,
        })
    }

    pub fn add_diacritics(&mut self, text: &str) -> Result<String> {
        self.add_diacritics_with_options(text, None)
    }
    
    pub fn add_diacritics_with_options(&mut self, text: &str, mark_matres_lectionis: Option<&str>) -> Result<String> {
        self.inner.run_inference(text, mark_matres_lectionis)
    }
}
