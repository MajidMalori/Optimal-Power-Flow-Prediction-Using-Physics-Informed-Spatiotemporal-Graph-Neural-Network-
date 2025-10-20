# Fixes Summary - Power System ML Project

## ✅ Fixed: Carbon Emissions Calculation

### **Problem Identified**
- Carbon emissions were **NEGATIVE** (-0.855 tCO2 for 33-bus, -1,765 tCO2 for 57-bus)
- MOOPF scores were negative due to negative carbon component

### **Root Cause**
The carbon emissions formula was using **total generation** (slack bus + renewables) instead of **only renewable generation**:
```python
# OLD (WRONG):
total_distributed_gen = total_gen  # Includes slack bus!
power_from_grid = total_load - total_distributed_gen  # Often NEGATIVE!
```

### **Solution Implemented**
Now correctly uses **only renewable generation** based on the renewable_fraction metadata:
```python
# NEW (CORRECT):
total_distributed_gen = total_load * renewable_fraction  # Renewables only
power_from_grid = total_load - total_distributed_gen  # Always positive!
```

### **Files Modified**
1. `utils/metrics.py` - Updated `_compute_carbon_emissions()` to accept `renewable_fraction` parameter
2. `utils/evaluation.py` - Pass `renewable_fraction` from batch data to emissions calculation

### **Expected Results**
- ✅ Carbon emissions will now be **positive** for all systems
- ✅ Higher renewable fractions → lower emissions (physically correct)
- ✅ MOOPF scores will be positive and meaningful

---

## 🔄 Next: Ybus Memory Optimization (Pending)

### **Current Problem**
- **Ybus storage**: 2.1 GB per scenario × 6 scenarios = **12.6 GB total**
- **95% redundant**: Only ~5% of timesteps have topology changes (N-1 contingencies)
- **Memory error on 118-bus**: Cannot load 60,000 Ybus matrices

### **Proposed Solution: Option 2 (Base + Sparse Differences)**

#### **Strategy:**
1. Store 1 base Ybus per renewable fraction (6 total)
2. Store timestep indices where contingencies happened
3. Store only the changed Ybus matrices for those timesteps

#### **Storage Savings:**
```
Current:  60,000 Ybus matrices × 222 KB = 13.3 GB
Proposed: 6 base + ~3,000 contingency = 0.67 GB
Savings:  12.6 GB (95% reduction!)
```

#### **Implementation Plan:**
1. **Data Generation** (`data/gen_meas_best.py`):
   - Track which timesteps have contingencies
   - Save base Ybus + contingency_indices + contingency_ybus separately
   
2. **Data Loading** (`utils/data_loader.py`):
   - Load base Ybus
   - Load contingency indices and matrices
   - Reconstruct full Ybus array by filling base where no contingency

3. **Backward Compatibility**:
   - Check if loading new format (small file) or old format (large file)
   - Auto-detect and handle both

---

## 📊 Testing Plan

### **1. Test Carbon Emissions Fix**
Run training on all systems:
```bash
python train.py
```

**Expected Output:**
- ✅ Positive carbon emissions for all bus systems
- ✅ Emissions decrease with higher renewable fractions
- ✅ MOOPF scores are positive

### **2. Test Ybus Optimization** (After carbon fix verified)
```bash
# Regenerate data with new Ybus format
cd data
python gen_meas_best.py

# Run training (should work without memory errors)
cd ..
python train.py
```

---

## 🐛 Debug Prints (Temporary)

Added debug prints in `utils/metrics.py` to verify calculations:
- Total load, total generation, renewable generation
- Power from grid, carbon intensity, energy coefficients
- Raw and normalized emissions

**TODO: Remove debug prints after verification**

---

## 📈 Performance Issues to Address

### **57-Bus and 118-Bus MSE**
- **33-bus MSE**: 0.000125 ✅ (excellent)
- **57-bus MSE**: 2,572,142 ❌ (20 million times worse!)
- **118-bus**: Not tested yet (memory error)

### **Possible Causes:**
1. **Scale mismatch**: Larger systems may need different normalization
2. **Loss weight imbalance**: Physics losses dominating for larger systems
3. **Model capacity**: Same model size for all systems (may need scaling)
4. **Learning rate**: May need adjustment for larger systems

### **Investigation Needed:**
- Check if MSE is in denormalized or normalized units
- Compare physics violation magnitudes across systems
- Analyze gradients during training
- Test with per-system hyperparameters

---

## 🎯 Priority Order

1. ✅ **DONE**: Fix carbon emissions calculation
2. **NEXT**: Test carbon fix and verify results
3. **THEN**: Implement Ybus optimization (if memory is issue)
4. **FINALLY**: Debug 57-bus/118-bus performance degradation

