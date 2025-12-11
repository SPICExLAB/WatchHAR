import numpy as np
import scipy

def get_mel(sr, n_fft, n_mels=128, fmin=0.0, fmax=None, htk=False, norm=1):
    """Create a Mel filter bank matrix."""
    if fmax is None:
        fmax = float(sr) / 2
    
    # Create mel frequency points
    mel_f = mel_frequencies(n_mels + 2, fmin=fmin, fmax=fmax, htk=htk)
    
    # Convert mel to Hz
    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fftfreqs(sr, n_fft))
    
    for i in range(n_mels):
        # lower and upper slopes for all bins
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i+2] / fdiff[i+1]
        
        # Intersect them with each other and zero
        weights = np.maximum(0, np.minimum(lower, upper))
        
        if norm == 1:
            # Slaney-style mel is scaled to be approx constant energy per channel
            enorm = 2.0 / (mel_f[i+2] - mel_f[i])
            weights *= enorm
        
        if i == 0:
            mel_basis = weights.reshape(1, -1)
        else:
            mel_basis = np.vstack([mel_basis, weights])
    
    return mel_basis

def mel_frequencies(n_mels=128, fmin=0.0, fmax=11025.0, htk=False):
    """Compute an array of acoustic frequencies tuned to the mel scale."""
    # 'Center freqs' of mel bands - uniformly spaced between limits
    min_mel = hz_to_mel(fmin, htk=htk)
    max_mel = hz_to_mel(fmax, htk=htk)
    
    mels = np.linspace(min_mel, max_mel, n_mels)
    
    return mel_to_hz(mels, htk=htk)

def hz_to_mel(frequencies, htk=False):
    """Convert Hz to Mels."""
    frequencies = np.asanyarray(frequencies)
    
    if htk:
        return 2595.0 * np.log10(1.0 + frequencies / 700.0)
    
    # Fill in the linear part
    f_min = 0.0
    f_sp = 200.0 / 3
    
    mels = (frequencies - f_min) / f_sp
    
    # Fill in the log-scale part
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = np.log(6.4) / 27.0
    
    if frequencies.ndim:
        # If we have array data, vectorize
        log_t = (frequencies >= min_log_hz)
        mels[log_t] = min_log_mel + np.log(frequencies[log_t]/min_log_hz) / logstep
    elif frequencies >= min_log_hz:
        # If we have scalar data, heck directly
        mels = min_log_mel + np.log(frequencies / min_log_hz) / logstep
    
    return mels

def mel_to_hz(mels, htk=False):
    """Convert mel bin numbers to frequencies."""
    mels = np.asanyarray(mels)
    
    if htk:
        return 700.0 * (10.0**(mels / 2595.0) - 1.0)
    
    # Fill in the linear scale
    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mels
    
    # And now the nonlinear scale
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = np.log(6.4) / 27.0
    
    if mels.ndim:
        # If we have vector data, vectorize
        log_t = (mels >= min_log_mel)
        freqs[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))
    elif mels >= min_log_mel:
        # If we have scalar data, check directly
        freqs = min_log_hz * np.exp(logstep * (mels - min_log_mel))
    
    return freqs

def fftfreqs(sr=22050, n_fft=2048):
    """Alternative implementation of `np.fft.fftfreq`."""
    return np.linspace(0, float(sr) / 2, int(1 + n_fft//2), endpoint=True)

def pad_center(data, size, axis=-1, **kwargs):
    """Wrapper for np.pad to automatically center an array prior to padding."""
    kwargs.setdefault('mode', 'constant')
    
    n = data.shape[axis]
    
    lpad = int((size - n) // 2)
    
    lengths = [(0, 0)] * data.ndim
    lengths[axis] = (lpad, int(size - n - lpad))
    
    if lpad < 0:
        raise ValueError(('Target size ({:d}) must be '
                         'at least input size ({:d})').format(size, n))
    
    return np.pad(data, lengths, **kwargs)